// Comparison benchmark: fastcdc-rs (v2020, 4.x) vs plain mincdc vs mincdc with
// the caterpillar layer (tier-1 RLE of byte-identical adjacent chunks), across
// scenarios that stress speed, metadata (record count) and deduplication.
//
//   cargo run --release --example compare
//
// Metrics per algorithm:
//   GB/s     chunking throughput (boundary production only, no hashing)
//   records  number of chunk/segment records (== metadata rows)
//   mean     mean chunk size in bytes
//   uniq     distinct records after content addressing
//   dedup%   1 - stored_bytes / logical_bytes  (stored = sum of unique record sizes)
//
// Note: plain CDC already dedups identical bytes (uniq/dedup%), so the
// caterpillar win shows up in *records* (metadata), not dedup% — except where
// the periodic RLE genuinely stores fewer bytes.

use std::time::Instant;

use fastcdc::v2020::FastCDC;
use mincatcdc::caterpillar::CaterpillarChunker;
use mincatcdc::{MinCdcHash4, SliceChunker};

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[derive(Default)]
struct Stats {
    records: usize,
    logical: usize,
    gbps: f64,
}

/// Accumulates (key, stored_len) records and reports unique/dedup over them.
struct Dedup {
    seen: std::collections::HashMap<u64, usize>,
    logical: usize,
    records: usize,
}
impl Dedup {
    fn new() -> Self {
        Self {
            seen: std::collections::HashMap::new(),
            logical: 0,
            records: 0,
        }
    }
    fn add(&mut self, key: u64, stored_len: usize, logical_len: usize) {
        self.seen.entry(key).or_insert(stored_len);
        self.logical += logical_len;
        self.records += 1;
    }
    fn unique(&self) -> usize {
        self.seen.len()
    }
    fn dedup_pct(&self) -> f64 {
        let stored: usize = self.seen.values().sum();
        100.0 * (1.0 - stored as f64 / self.logical.max(1) as f64)
    }
}

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

/// English-ish low-entropy text: a small word pool joined with spaces.
fn texty(seed: u64, n: usize) -> Vec<u8> {
    let words: [&[u8]; 16] = [
        b"the", b"quick", b"brown", b"fox", b"jumps", b"over", b"lazy", b"dog", b"and", b"then",
        b"runs", b"away", b"into", b"dark", b"deep", b"woods",
    ];
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n + 16);
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(words[(s as usize) & 15]);
        out.push(b' ');
    }
    out.truncate(n);
    out
}

// ---- per-algorithm chunking + analysis -------------------------------------

const ITERS: usize = 3;

fn bench<F: FnMut() -> usize>(bytes_len: usize, mut produce: F) -> (f64, usize) {
    let mut best = f64::MAX;
    let mut records = 0;
    for _ in 0..ITERS {
        let t = Instant::now();
        records = produce();
        let e = t.elapsed().as_secs_f64();
        best = best.min(e);
    }
    let gbps = bytes_len as f64 / best / 1e9;
    (gbps, records)
}

fn run_fastcdc(data: &[u8], min: usize, avg: usize, max: usize, d: &mut Dedup) -> Stats {
    let (gbps, records) = bench(data.len(), || {
        let mut n = 0;
        for c in FastCDC::new(data, min, avg, max) {
            std::hint::black_box(c.offset);
            n += 1;
        }
        n
    });
    for c in FastCDC::new(data, min, avg, max) {
        let b = &data[c.offset..c.offset + c.length];
        d.add(fnv1a(b), b.len(), b.len());
    }
    Stats {
        records,
        logical: data.len(),
        gbps,
    }
}

fn run_mincdc(data: &[u8], min: usize, max: usize, d: &mut Dedup) -> Stats {
    let cdc = MinCdcHash4::new();
    let (gbps, records) = bench(data.len(), || {
        let mut n = 0;
        for c in SliceChunker::new(data, min, max, cdc) {
            std::hint::black_box(c.offset());
            n += 1;
        }
        n
    });
    for c in SliceChunker::new(data, min, max, cdc) {
        d.add(fnv1a(&c), c.len(), c.len());
    }
    Stats {
        records,
        logical: data.len(),
        gbps,
    }
}

fn run_caterpillar(data: &[u8], min: usize, max: usize, d: &mut Dedup) -> Stats {
    let cdc = MinCdcHash4::new();
    let make = || CaterpillarChunker::new(data, min, max, cdc);
    let (gbps, records) = bench(data.len(), || {
        let mut n = 0;
        for s in make() {
            std::hint::black_box(s.offset());
            n += 1;
        }
        n
    });
    for s in make() {
        d.add(fnv1a(s.dedup_key()), s.dedup_key().len(), s.len());
    }
    Stats {
        records,
        logical: data.len(),
        gbps,
    }
}

fn row(name: &str, s: &Stats, d: &Dedup) {
    let mean = s.logical.checked_div(s.records).unwrap_or(0);
    println!(
        "  {name:<18} {gbps:>7.2} GB/s  records={rec:>7}  mean={mean:>6}  uniq={uniq:>7}  dedup={dd:>5.1}%",
        gbps = s.gbps,
        rec = s.records,
        uniq = d.unique(),
        dd = d.dedup_pct(),
    );
}

/// Versioned dedup: chunk v1 and v2 (= v1 with an inserted blob), union the
/// records, report cross-version dedup. Shows the caterpillar layer doesn't hurt
/// normal CDC dedup.
fn scenario_versioned(min: usize, avg: usize, max: usize) {
    let v1 = xorshift(2024, 8 * 1024 * 1024);
    let mut v2 = Vec::with_capacity(v1.len() + 5000);
    let cut = v1.len() / 2;
    v2.extend_from_slice(&v1[..cut]);
    v2.extend_from_slice(&xorshift(999, 5000)); // inserted edit
    v2.extend_from_slice(&v1[cut..]);
    let total = v1.len() + v2.len();
    println!(
        "\n=== versioned (v1 + v2-with-insert)  ({} MiB total) ===",
        total / (1024 * 1024)
    );

    // fastcdc
    let mut d = Dedup::new();
    for data in [&v1[..], &v2[..]] {
        for c in FastCDC::new(data, min, avg, max) {
            let b = &data[c.offset..c.offset + c.length];
            d.add(fnv1a(b), b.len(), b.len());
        }
    }
    println!(
        "  {:<18} records={:>7}  uniq={:>7}  dedup={:>5.1}%",
        "fastcdc-v2020",
        d.records,
        d.unique(),
        d.dedup_pct()
    );

    // mincdc plain
    let cdc = MinCdcHash4::new();
    let mut d = Dedup::new();
    for data in [&v1[..], &v2[..]] {
        for c in SliceChunker::new(data, min, max, cdc) {
            d.add(fnv1a(&c), c.len(), c.len());
        }
    }
    println!(
        "  {:<18} records={:>7}  uniq={:>7}  dedup={:>5.1}%",
        "mincdc-plain",
        d.records,
        d.unique(),
        d.dedup_pct()
    );

    // mincdc + caterpillar
    let mut d = Dedup::new();
    for data in [&v1[..], &v2[..]] {
        for s in CaterpillarChunker::new(data, min, max, cdc) {
            d.add(fnv1a(s.dedup_key()), s.dedup_key().len(), s.len());
        }
    }
    println!(
        "  {:<18} records={:>7}  uniq={:>7}  dedup={:>5.1}%",
        "mincdc+cat",
        d.records,
        d.unique(),
        d.dedup_pct()
    );
}

fn main() {
    // ~8 KiB target. Wide window (normal CDC). fastcdc gets a large max to avoid
    // frequent forced cuts (its long tail), per the mincdc README methodology.
    let (min, avg, max) = (2048usize, 8192usize, 14336usize);
    let fast_max = avg + (avg - min) * 7; // 51200
    let n = 16 * 1024 * 1024;
    run_suite(min, avg, max, fast_max, n);
}

fn run_suite(min: usize, avg: usize, mc_max: usize, fast_max: usize, n: usize) {
    // helper that uses mincdc max for mincdc/caterpillar and fast_max for fastcdc
    let block = |title: &str, data: &[u8]| {
        println!("\n=== {title}  ({} MiB) ===", data.len() / (1024 * 1024));
        let mut d = Dedup::new();
        let s = run_fastcdc(data, min, avg, fast_max, &mut d);
        row("fastcdc-v2020", &s, &d);
        let mut d = Dedup::new();
        let s = run_mincdc(data, min, mc_max, &mut d);
        row("mincdc-plain", &s, &d);
        let mut d = Dedup::new();
        let s = run_caterpillar(data, min, mc_max, &mut d);
        row("mincdc+cat", &s, &d);
    };

    block("random / incompressible", &xorshift(1, n));
    block("zero-fill", &vec![0u8; n]);
    block("english-ish text", &texty(5, n));

    // Repeated fixed-size record (period 777), wide window: mincdc self-aligns.
    {
        let p = xorshift(42, 777);
        let mut data = Vec::with_capacity(n);
        while data.len() < n {
            data.extend_from_slice(&p);
        }
        block("periodic-777 (wide win)", &data);
    }

    // Same data, NARROW window where no period multiple fits: tier-1 cannot
    // coalesce the rotating chunks (this is the case the removed period-detection
    // tier targeted; see examples/CATBENCH_RESULTS.md for why it wasn't worth it).
    {
        let p = xorshift(42, 777);
        let mut data = Vec::with_capacity(n);
        while data.len() < n {
            data.extend_from_slice(&p);
        }
        let (nmin, nmax) = (2048usize, 2200usize);
        println!(
            "\n=== periodic-777 (NARROW win min={nmin} max={nmax})  ({} MiB) ===",
            data.len() / (1024 * 1024)
        );
        let mut d = Dedup::new();
        let s = run_fastcdc(&data, nmin, 2100, nmax, &mut d);
        row("fastcdc-v2020", &s, &d);
        let mut d = Dedup::new();
        let s = run_mincdc(&data, nmin, nmax, &mut d);
        row("mincdc-plain", &s, &d);
        let mut d = Dedup::new();
        let s = run_caterpillar(&data, nmin, nmax, &mut d);
        row("mincdc+cat", &s, &d);
    }

    // Mixed: random with embedded zero-runs and a repeated record block.
    {
        let mut data = xorshift(7, n);
        // a big zero hole
        for b in data.iter_mut().take(n / 2).skip(n / 4) {
            *b = 0;
        }
        // a repeated-record region near the end
        let rec = xorshift(3, 333);
        let start = n - n / 8;
        let mut i = start;
        while i + rec.len() <= n {
            data[i..i + rec.len()].copy_from_slice(&rec);
            i += rec.len();
        }
        block("mixed (random+holes+recs)", &data);
    }

    scenario_versioned(min, avg, fast_max);
}
