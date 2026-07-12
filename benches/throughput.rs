//! Criterion throughput comparison for the caterpillar packed-scanning fast
//! path:
//!
//!   plain               SliceChunker, boundaries only (the Phase 1 baseline).
//!   caterpillar-scalar  the pre-SIMD caterpillar, inlined here as the
//!                       baseline: pull every chunk (a full argmin boundary
//!                       search each) and RLE-coalesce byte-identical
//!                       neighbors.
//!   caterpillar-packed  the shipped MothChunker: one packed equality
//!                       scan proves a run periodic and emits the repeats
//!                       without running the boundary search for them.
//!
//! Run with `cargo bench`. No RUSTFLAGS are needed: the SIMD paths are built
//! with `#[target_feature]` and chosen by runtime CPU detection, so this
//! measures the best width the machine has (NEON on aarch64; SSE2/AVX2/
//! AVX-512BW on x86_64).
//!
//! The synthetic set spans best case (`zeros`: every byte in one run, packed
//! scanning reduces to a memory-bandwidth compare — the theoretical ceiling),
//! structurally realistic mixes (`disk-image-like`, `padded-binary-like`),
//! and honest no-op cases where the caterpillar cannot help (`random`,
//! `log-like`: repetitive but never byte-identical at chunk granularity).
//!
//! To bench real files instead, set `BENCH_CORPUS=/path/to/dir` — every
//! regular file in the directory becomes a dataset (name = file name).

use std::path::Path;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mothcdc::MothChunker;
use mothcdc::mincdc::{MinCdcHash4, SliceChunker};

const MIN: usize = 4096;
const MAX: usize = 12288;
const SIZE: usize = 8 * 1024 * 1024;

fn xorshift(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 33) as u8
        })
        .collect()
}

/// Deterministic usize in `0..bound` from an evolving seed.
fn next_rand(s: &mut u64, bound: usize) -> usize {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 16) as usize % bound
}

/// Mostly-empty filesystem image shape: random "metadata/file" clusters
/// scattered through zero free space (~70% zeros in 4 KiB..512 KiB spans,
/// mirroring what a written-then-emptied disk image looks like).
fn disk_image_like() -> Vec<u8> {
    let mut data = vec![0u8; SIZE];
    let mut s = 0xD15Cu64;
    let mut pos = 0usize;
    while pos < SIZE {
        // Zero span (free space), then a data cluster (file contents).
        pos += 4096 + next_rand(&mut s, 512 * 1024);
        if pos >= SIZE {
            break;
        }
        let cluster = 4096 + next_rand(&mut s, 128 * 1024);
        let end = (pos + cluster).min(SIZE);
        let content = xorshift(s, end - pos);
        data[pos..end].copy_from_slice(&content);
        pos = end;
    }
    data
}

/// Executable/object-file shape: dense sections separated by page-aligned
/// zero padding of 4..64 KiB (linkers pad sections; rlibs pad members).
fn padded_binary_like() -> Vec<u8> {
    let mut data = Vec::with_capacity(SIZE);
    let mut s = 0xB1Au64;
    while data.len() < SIZE {
        let section = 16 * 1024 + next_rand(&mut s, 256 * 1024);
        data.extend_from_slice(&xorshift(s, section));
        let pad = 4096 + next_rand(&mut s, 60 * 1024);
        data.resize((data.len() + pad).min(SIZE), 0);
    }
    data.truncate(SIZE);
    data
}

/// Log-stream shape: one template line repeated with a changing timestamp,
/// sequence number, and id. Highly compressible and repetitive to a human,
/// but chunks are never byte-identical — the caterpillar (old or new) must
/// be a no-op here. This is the honesty case.
fn log_like() -> Vec<u8> {
    let mut data = Vec::with_capacity(SIZE + 256);
    let mut s = 0x106u64;
    let mut seq = 0u64;
    while data.len() < SIZE {
        seq += 1;
        let ts = 1_752_000_000_000u64 + seq * 37 + next_rand(&mut s, 25) as u64;
        let line = format!(
            "{{\"ts\":{ts},\"level\":\"INFO\",\"service\":\"api-gateway\",\"seq\":{seq},\
             \"msg\":\"request completed\",\"path\":\"/v1/objects/{:08x}\",\"status\":200,\
             \"latency_ms\":{}}}\n",
            next_rand(&mut s, u32::MAX as usize),
            next_rand(&mut s, 900),
        );
        data.extend_from_slice(line.as_bytes());
    }
    data.truncate(SIZE);
    data
}

fn datasets() -> Vec<(String, Vec<u8>)> {
    // Real-file corpus mode: every regular file in $BENCH_CORPUS is a dataset.
    if let Ok(dir) = std::env::var("BENCH_CORPUS") {
        let mut out: Vec<(String, Vec<u8>)> = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(Path::new(&dir))
            .expect("BENCH_CORPUS must be a readable directory")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file())
            .collect();
        entries.sort();
        for p in entries {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&p).expect("corpus file must be readable");
            if !bytes.is_empty() {
                out.push((name, bytes));
            }
        }
        assert!(
            !out.is_empty(),
            "BENCH_CORPUS directory has no usable files"
        );
        return out;
    }

    let mut periodic = Vec::with_capacity(SIZE + 777);
    let unit = xorshift(9, 777);
    while periodic.len() < SIZE {
        periodic.extend_from_slice(&unit);
    }
    periodic.truncate(SIZE);

    vec![
        ("zeros".into(), vec![0u8; SIZE]),
        ("periodic-777".into(), periodic),
        ("disk-image-like".into(), disk_image_like()),
        ("padded-binary-like".into(), padded_binary_like()),
        ("log-like".into(), log_like()),
        ("random".into(), xorshift(1, SIZE)),
    ]
}

/// The pre-SIMD caterpillar: RLE-coalesce over the plain chunker, paying a
/// full boundary search for every chunk in a run. Returns the record count.
fn caterpillar_scalar_reference(data: &[u8]) -> usize {
    let mut records = 0usize;
    let mut last: Option<(usize, usize)> = None; // (offset, len) of current unit
    for c in SliceChunker::new(data, MIN, MAX, MinCdcHash4::new()) {
        match last {
            Some((off, len)) if data[off..off + len] == c[..] => {},
            _ => {
                records += 1;
                last = Some((c.offset(), c.len()));
            },
        }
    }
    records
}

fn bench_chunking(c: &mut Criterion) {
    // Real files can be hundreds of MiB; fewer samples keep runs tractable.
    let corpus_mode = std::env::var("BENCH_CORPUS").is_ok();
    let mut g = c.benchmark_group("chunking");
    g.sample_size(if corpus_mode { 10 } else { 30 });

    for (name, data) in datasets() {
        g.throughput(Throughput::Bytes(data.len() as u64));

        g.bench_with_input(BenchmarkId::new("plain", &name), &data, |b, d| {
            b.iter(|| {
                let mut acc = 0usize;
                for c in SliceChunker::new(d, MIN, MAX, MinCdcHash4::new()) {
                    acc ^= c.offset();
                }
                acc
            })
        });

        g.bench_with_input(
            BenchmarkId::new("caterpillar-scalar", &name),
            &data,
            |b, d| b.iter(|| caterpillar_scalar_reference(d)),
        );

        g.bench_with_input(
            BenchmarkId::new("caterpillar-packed", &name),
            &data,
            |b, d| {
                b.iter(|| {
                    let mut acc = 0usize;
                    for s in MothChunker::new(d, MIN, MAX) {
                        acc ^= s.offset() ^ s.chunk_count();
                    }
                    acc
                })
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench_chunking);
criterion_main!(benches);
