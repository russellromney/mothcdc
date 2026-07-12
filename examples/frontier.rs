// Frontier sweep: how (min_size, max_size) and the MinCDCHash4 parameters
// move the three quantities that matter — throughput, dedup savings, and
// metadata records — on a real corpus.
//
//   cargo run --release --example frontier -- <path> [<path> ...]
//
// For each configuration this chunks the whole corpus with the caterpillar
// and reports:
//   GB/s     chunking throughput (best of 3, boundaries + coalescing only)
//   savings  1 - unique dedup_key bytes / total bytes (fnv-keyed, like a
//            content-addressed store would see)
//   records  caterpillar metadata records
//   chunks   underlying mincdc chunks (records before coalescing)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mincatcdc::{CaterpillarChunker, MinCdcHash4};

fn collect_files(root: &Path, out: &mut Vec<PathBuf>) {
    if root.is_dir() {
        let mut e: Vec<_> = std::fs::read_dir(root)
            .map(|rd| rd.filter_map(|x| x.ok().map(|x| x.path())).collect())
            .unwrap_or_default();
        e.sort();
        for p in e {
            collect_files(&p, out);
        }
    } else if root.is_file() {
        out.push(root.to_path_buf());
    }
}

fn fnv(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn run(label: &str, blobs: &[Vec<u8>], total: u64, min: usize, max: usize, cdc: MinCdcHash4) {
    // Throughput: best of 3, boundaries + coalescing only.
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let mut acc = 0usize;
        for b in blobs {
            for s in CaterpillarChunker::new(b, min, max, cdc) {
                acc ^= s.offset();
            }
        }
        std::hint::black_box(acc);
        best = best.min(t.elapsed().as_secs_f64());
    }

    // Dedup + metadata (single deterministic pass).
    let mut store: HashMap<u64, usize> = HashMap::new();
    let (mut records, mut chunks) = (0usize, 0usize);
    for b in blobs {
        for s in CaterpillarChunker::new(b, min, max, cdc) {
            let key = s.dedup_key();
            store.entry(fnv(key)).or_insert(key.len());
            records += 1;
            chunks += s.chunk_count();
        }
    }
    let unique: usize = store.values().sum();
    let savings = 100.0 * (1.0 - unique as f64 / total as f64);
    println!(
        "  {label:<28} {gbps:>6.2} GB/s  savings={savings:>6.3}%  records={records:>8}  chunks={chunks:>8}",
        gbps = total as f64 / best / 1e9,
    );
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let mut files = Vec::new();
    for p in &paths {
        collect_files(Path::new(p), &mut files);
    }
    let mut blobs = Vec::new();
    let mut total: u64 = 0;
    for f in &files {
        if let Ok(b) = std::fs::read(f) {
            total += b.len() as u64;
            blobs.push(b);
        }
    }
    println!(
        "### {} files, {:.1} MiB",
        blobs.len(),
        total as f64 / (1024.0 * 1024.0)
    );

    println!("--- (min,max) sweep, default hash ---");
    for (min, max) in [
        (1024usize, 8192usize),
        (2048, 8192),
        (2048, 14336),
        (2048, 32768),
        (4096, 12288),
        (4096, 16384),
        (4096, 65536),
        (8192, 24576),
    ] {
        run(
            &format!("min={min} max={max}"),
            &blobs,
            total,
            min,
            max,
            MinCdcHash4::new(),
        );
    }

    println!("--- hash-parameter sweep at (2048, 14336) ---");
    for (name, m, a) in [
        ("default (0x915f77f5)", 0x915f77f5u32, 0x34636463u32),
        ("golden  (0x9e3779b1)", 0x9e3779b1, 0x34636463),
        ("murmur3 (0xcc9e2d51)", 0xcc9e2d51, 0x1b873593),
        ("mul-only (a=0)", 0x915f77f5, 0),
        ("low-entropy M (0x3)", 0x3, 0x34636463),
    ] {
        run(
            name,
            &blobs,
            total,
            2048,
            14336,
            MinCdcHash4::with_params(m, a),
        );
    }
}
