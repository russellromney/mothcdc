// Focused speed / latency / metadata comparison: plain mincdc vs the caterpillar
// layers (tier-1 RLE, tier-2 period). No fastcdc. Best-of-N timing.
//
//   cargo run --release --example catbench -- <path> [<path> ...]
//
// Reports, per algorithm:
//   GB/s     chunking throughput (best of N, boundaries only)
//   ms       wall time to chunk the whole corpus
//   ns/chunk per-record latency
//   records  metadata rows
//   metaMiB  records * 48 bytes (32B hash + offset/len), the metadata footprint
//   meta%    metadata bytes / corpus bytes

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mincatcdc::caterpillar::{CaterpillarChunker, Segment};
use mincatcdc::{MinCdcHash4, SliceChunker};

const MIN: usize = 2048;
const MAX: usize = 14336;
const ITERS: usize = 5;
const RECORD_BYTES: usize = 48; // 32B fingerprint + offset/len/bookkeeping

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

fn best_of<F: FnMut() -> usize>(mut f: F) -> (f64, usize) {
    let mut best = f64::MAX;
    let mut n = 0;
    for _ in 0..ITERS {
        let t = Instant::now();
        n = f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best, n)
}

fn report(name: &str, secs: f64, records: usize, total_bytes: u64, plain_rec: usize, plain_gbps: f64) {
    let gbps = total_bytes as f64 / secs / 1e9;
    let ns_chunk = secs * 1e9 / records.max(1) as f64;
    let meta = records * RECORD_BYTES;
    let meta_mib = meta as f64 / (1024.0 * 1024.0);
    let meta_pct = 100.0 * meta as f64 / total_bytes.max(1) as f64;
    let speed_delta = if plain_gbps > 0.0 {
        format!("{:+.0}%", 100.0 * (gbps / plain_gbps - 1.0))
    } else {
        "—".into()
    };
    let rec_delta = if plain_rec > 0 {
        format!("{:+.0}%", 100.0 * (records as f64 / plain_rec as f64 - 1.0))
    } else {
        "—".into()
    };
    println!(
        "  {name:<14} {gbps:>6.2} GB/s ({sd:>5})  {ms:>7.1} ms  {nc:>6.1} ns/chunk  \
         rec={rec:>8} ({rd:>5})  meta={mm:>6.1} MiB  meta%={mp:>4.2}",
        sd = speed_delta,
        ms = secs * 1e3,
        nc = ns_chunk,
        rec = records,
        rd = rec_delta,
        mm = meta_mib,
        mp = meta_pct,
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
    println!("\n### {} files, {:.1} MiB", blobs.len(), total as f64 / (1024.0 * 1024.0));
    let cdc = MinCdcHash4::new();

    // also report unique records (dedup) once, deterministic.
    let dedup = |full: Option<bool>| -> (usize, usize) {
        // returns (records, unique)
        let mut seen: HashMap<u64, ()> = HashMap::new();
        let mut rec = 0;
        let fnv = |b: &[u8]| {
            let mut h: u64 = 0xcbf29ce484222325;
            for &x in b {
                h ^= x as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };
        for b in &blobs {
            match full {
                None => {
                    for c in SliceChunker::new(b, MIN, MAX, cdc) {
                        seen.insert(fnv(&c), ());
                        rec += 1;
                    }
                }
                Some(f) => {
                    let it = if f {
                        CaterpillarChunker::new(b, MIN, MAX, cdc).with_period_detection(usize::MAX)
                    } else {
                        CaterpillarChunker::new(b, MIN, MAX, cdc)
                    };
                    for s in it {
                        let k = match s {
                            Segment::Solo(c) => fnv(&c),
                            Segment::Caterpillar { unit, .. } => fnv(unit),
                            Segment::Periodic { ref canonical, .. } => fnv(canonical),
                        };
                        seen.insert(k, ());
                        rec += 1;
                    }
                }
            }
        }
        (rec, seen.len())
    };

    // plain
    let (secs, _) = best_of(|| {
        let mut n = 0usize;
        let mut s = 0usize;
        for b in &blobs {
            for c in SliceChunker::new(b, MIN, MAX, cdc) {
                s ^= c.offset();
                n += 1;
            }
        }
        std::hint::black_box(s);
        n
    });
    let (prec, puniq) = dedup(None);
    let plain_gbps = total as f64 / secs / 1e9;
    report("mincdc-plain", secs, prec, total, prec, plain_gbps);
    println!("                 (unique records: {puniq})");

    for (name, full) in [("cat-simple", false), ("cat-period", true)] {
        let (secs, _) = best_of(|| {
            let mut n = 0usize;
            let mut s = 0usize;
            for b in &blobs {
                let it = if full {
                    CaterpillarChunker::new(b, MIN, MAX, cdc).with_period_detection(usize::MAX)
                } else {
                    CaterpillarChunker::new(b, MIN, MAX, cdc)
                };
                for seg in it {
                    s ^= seg.offset();
                    n += 1;
                }
            }
            std::hint::black_box(s);
            n
        });
        let (rec, uniq) = dedup(Some(full));
        report(name, secs, rec, total, prec, plain_gbps);
        println!("                 (unique records: {uniq})");
    }
}
