//! Criterion throughput comparison for the caterpillar packed-scanning fast
//! path:
//!
//!   plain               SliceChunker, boundaries only (the Phase 1 baseline).
//!   caterpillar-scalar  the pre-SIMD caterpillar, inlined here as the
//!                       baseline: pull every chunk (a full argmin boundary
//!                       search each) and RLE-coalesce byte-identical
//!                       neighbors.
//!   caterpillar-packed  the shipped CaterpillarChunker: one packed equality
//!                       scan proves a run periodic and emits the repeats
//!                       without running the boundary search for them.
//!
//! Run with `cargo bench`. No RUSTFLAGS are needed: the SIMD paths are built
//! with `#[target_feature]` and chosen by runtime CPU detection, so this
//! measures the best width the machine has (NEON on aarch64; SSE2/AVX2/
//! AVX-512BW on x86_64).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mincatcdc::{CaterpillarChunker, MinCdcHash4, SliceChunker};

const MIN: usize = 2048;
const MAX: usize = 14336;
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

fn datasets() -> Vec<(&'static str, Vec<u8>)> {
    let mut periodic = Vec::with_capacity(SIZE + 777);
    let unit = xorshift(9, 777);
    while periodic.len() < SIZE {
        periodic.extend_from_slice(&unit);
    }
    periodic.truncate(SIZE);

    let mut holed = xorshift(2, SIZE);
    for b in &mut holed[SIZE / 4..3 * SIZE / 4] {
        *b = 0;
    }

    vec![
        ("zeros", vec![0u8; SIZE]),
        ("periodic-777", periodic),
        ("random", xorshift(1, SIZE)),
        ("random+zero-hole", holed),
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
    let mut g = c.benchmark_group("chunking");
    g.sample_size(30);

    for (name, data) in datasets() {
        g.throughput(Throughput::Bytes(data.len() as u64));

        g.bench_with_input(BenchmarkId::new("plain", name), &data, |b, d| {
            b.iter(|| {
                let mut acc = 0usize;
                for c in SliceChunker::new(d, MIN, MAX, MinCdcHash4::new()) {
                    acc ^= c.offset();
                }
                acc
            })
        });

        g.bench_with_input(
            BenchmarkId::new("caterpillar-scalar", name),
            &data,
            |b, d| b.iter(|| caterpillar_scalar_reference(d)),
        );

        g.bench_with_input(
            BenchmarkId::new("caterpillar-packed", name),
            &data,
            |b, d| {
                b.iter(|| {
                    let mut acc = 0usize;
                    for s in CaterpillarChunker::new(d, MIN, MAX, MinCdcHash4::new()) {
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
