// Real-corpus benchmark. Feed it files and/or directories; it recurses dirs,
// chunks every file with each algorithm, and reports throughput, record count
// (metadata), dedup, and the chunk-size distribution.
//
//   cargo run --release --example realbench -- <path> [<path> ...]
//
// Each file is chunked independently (boundaries never cross file edges, as in a
// real store); dedup is measured across the whole corpus (content addressing),
// so passing two version trees measures cross-version dedup.
//
// Algorithms: fastcdc-v2020 (4.x), mincdc-plain, mincdc+cat-simple (tier 1),
// mincdc+cat-period (tier 2). Target ~8 KiB; fastcdc gets a large max to avoid
// its forced-cut tail (per the mincdc README methodology).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use fastcdc::v2020::FastCDC;
use mincatcdc::caterpillar::{CaterpillarChunker, Segment};
use mincatcdc::{MinCdcHash4, SliceChunker};

const MIN: usize = 2048;
const AVG: usize = 8192;
const MC_MAX: usize = 14336;
fn fast_max() -> usize {
    AVG + (AVG - MIN) * 7
} // 51200

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn collect_files(root: &Path, out: &mut Vec<PathBuf>) {
    if root.is_dir() {
        let mut entries: Vec<_> = match std::fs::read_dir(root) {
            Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
            Err(_) => return,
        };
        entries.sort();
        for p in entries {
            collect_files(&p, out);
        }
    } else if root.is_file() {
        out.push(root.to_path_buf());
    }
}

struct Acc {
    records: usize,
    logical: u64,
    seen: HashMap<u64, usize>, // key -> stored bytes
    sizes: Vec<u32>,
    secs: f64,
}
impl Acc {
    fn new() -> Self {
        Self { records: 0, logical: 0, seen: HashMap::new(), sizes: Vec::new(), secs: 0.0 }
    }
    fn add(&mut self, key: u64, stored: usize, logical: usize) {
        self.seen.entry(key).or_insert(stored);
        self.records += 1;
        self.logical += logical as u64;
        self.sizes.push(logical as u32);
    }
    fn report(&self, name: &str, total_bytes: u64) {
        let gbps = total_bytes as f64 / self.secs.max(1e-9) / 1e9;
        let stored: u64 = self.seen.values().map(|&v| v as u64).sum();
        let dedup = 100.0 * (1.0 - stored as f64 / self.logical.max(1) as f64);
        let mut s = self.sizes.clone();
        s.sort_unstable();
        let pct = |q: f64| -> u32 {
            if s.is_empty() {
                0
            } else {
                s[((s.len() as f64 * q) as usize).min(s.len() - 1)]
            }
        };
        let mean = if self.records > 0 { self.logical / self.records as u64 } else { 0 };
        println!(
            "  {name:<18} {gbps:>6.2} GB/s  rec={rec:>8}  uniq={uniq:>8}  dedup={dd:>5.1}%  \
             mean={mean:>6}  p50={p50:>6}  p90={p90:>6}  p99={p99:>6}  max={mx:>7}",
            rec = self.records,
            uniq = self.seen.len(),
            dd = dedup,
            p50 = pct(0.50),
            p90 = pct(0.90),
            p99 = pct(0.99),
            mx = pct(1.0),
        );
    }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: realbench <path> [<path> ...]");
        std::process::exit(1);
    }
    let mut files = Vec::new();
    for p in &paths {
        collect_files(Path::new(p), &mut files);
    }
    // Load all files into memory once (so timing excludes disk I/O).
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    let mut total_bytes: u64 = 0;
    for f in &files {
        if let Ok(b) = std::fs::read(f) {
            total_bytes += b.len() as u64;
            blobs.push(b);
        }
    }
    println!(
        "\n### corpus: {} files, {:.1} MiB",
        blobs.len(),
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    let cdc = MinCdcHash4::new();

    // fastcdc-v2020
    let mut a = Acc::new();
    let t = Instant::now();
    let mut sink = 0usize;
    for b in &blobs {
        for c in FastCDC::new(b, MIN, AVG, fast_max()) {
            sink ^= c.offset;
        }
    }
    a.secs = t.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    for b in &blobs {
        for c in FastCDC::new(b, MIN, AVG, fast_max()) {
            let s = &b[c.offset..c.offset + c.length];
            a.add(fnv1a(s), s.len(), s.len());
        }
    }
    a.report("fastcdc-v2020", total_bytes);

    // mincdc-plain
    let mut a = Acc::new();
    let t = Instant::now();
    let mut sink = 0usize;
    for b in &blobs {
        for c in SliceChunker::new(b, MIN, MC_MAX, cdc) {
            sink ^= c.offset();
        }
    }
    a.secs = t.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    for b in &blobs {
        for c in SliceChunker::new(b, MIN, MC_MAX, cdc) {
            a.add(fnv1a(&c), c.len(), c.len());
        }
    }
    a.report("mincdc-plain", total_bytes);

    // cat-simple (tier 1) and cat-period (tier 2)
    for (label, full) in [("mincdc+cat-simple", false), ("mincdc+cat-period", true)] {
        let mut a = Acc::new();
        let t = Instant::now();
        let mut sink = 0usize;
        for b in &blobs {
            let it = if full {
                CaterpillarChunker::new(b, MIN, MC_MAX, cdc)
            } else {
                CaterpillarChunker::simple(b, MIN, MC_MAX, cdc)
            };
            for s in it {
                sink ^= s.offset();
            }
        }
        a.secs = t.elapsed().as_secs_f64();
        std::hint::black_box(sink);
        for b in &blobs {
            let it = if full {
                CaterpillarChunker::new(b, MIN, MC_MAX, cdc)
            } else {
                CaterpillarChunker::simple(b, MIN, MC_MAX, cdc)
            };
            for s in it {
                match s {
                    Segment::Solo(c) => a.add(fnv1a(&c), c.len(), c.len()),
                    Segment::Caterpillar { unit, count, .. } => {
                        a.add(fnv1a(unit), unit.len(), unit.len() * count)
                    }
                    Segment::Periodic { canonical, total_len, .. } => {
                        a.add(fnv1a(&canonical), canonical.len(), total_len)
                    }
                }
            }
        }
        a.report(label, total_bytes);
    }
}
