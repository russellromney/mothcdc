//! Caterpillar coalescing — a metadata-efficiency layer over the chunkers.
//!
//! Inspired by the "caterpillar" idea in the Chonkers algorithm (Berger, 2025):
//! periodic / low-entropy regions make any CDC emit a flood of tiny,
//! near-`min_size` chunks. Each chunk is a separate metadata record, so a long
//! zero-fill or repeated-record region costs metadata out of all proportion to
//! its information content.
//!
//! [`CaterpillarChunker`] wraps a [`SliceChunker`] and run-length-encodes
//! maximal runs of byte-identical adjacent chunks into a single
//! [`Segment::Caterpillar`] record (the unit + a repeat count). It catches
//! zero-fill, constant bytes, and repeated blocks; it is a no-op (one slice
//! compare per chunk) on data with no runs, so it keeps mincdc's speed and
//! deduplication everywhere else. The output is lossless.
//!
//! # Example
//! ```
//! use mincatcdc::{MinCdcHash4, caterpillar::{CaterpillarChunker, Segment}};
//!
//! let data = vec![0u8; 64 * 2048]; // a long zero run
//! let segs: Vec<_> = CaterpillarChunker::new(&data, 2048, 14336, MinCdcHash4::new()).collect();
//!
//! // The whole zero run collapses into one record instead of ~64 chunks.
//! assert_eq!(segs.len(), 1);
//! assert!(matches!(segs[0], Segment::Caterpillar { .. }));
//! // `dedup_key()` gives the unique bytes to fingerprint/store, regardless of variant.
//! assert!(segs[0].dedup_key().iter().all(|&b| b == 0));
//! ```
//!
//! ## Experimental: period detection
//! [`CaterpillarChunker::with_period_detection`] additionally catches
//! *phase-rotating* periodic runs (consecutive chunks that are rotations of the
//! same period, which the default cannot see as equal). It is usually not worth
//! it — it adds a large per-chunk cost and mincdc already self-aligns to most
//! periods — so it is opt-in. The detected period cell is canonicalized to its
//! least rotation (Booth's algorithm) so the same periodic content dedups
//! regardless of phase. See `examples/CATBENCH_RESULTS.md`.

use crate::{Cdc, Chunk, SliceChunker};

/// Largest period (bytes) tier-2 detection will look for. Bounds the gate scan
/// and keeps the common (random) path cheap.
pub const MAX_PERIOD: usize = 4096;

/// Length of the prefix used by the period-recurrence gate.
const PROBE: usize = 16;

/// One output unit of [`CaterpillarChunker`].
#[derive(Debug)]
pub enum Segment<'a> {
    /// A single chunk whose neighbor differed — emitted as-is.
    Solo(Chunk<'a>),
    /// A run of `count` (>= 2) byte-identical adjacent chunks (tier 1).
    Caterpillar {
        /// Start offset of the run within the input.
        offset: usize,
        /// The repeated chunk's bytes.
        unit: &'a [u8],
        /// Number of times `unit` repeats (>= 2).
        count: usize,
    },
    /// A `P`-periodic region spanning `chunks` (>= 2) underlying chunks (tier 2).
    /// Reconstructed by tiling `raw_period` across `total_len` bytes.
    Periodic {
        /// Start offset of the run within the input.
        offset: usize,
        /// The period as it appears at `offset` (length `P`).
        raw_period: &'a [u8],
        /// Total bytes covered (may not be a whole multiple of `P`).
        total_len: usize,
        /// Underlying chunk count represented.
        chunks: usize,
        /// Least rotation of `raw_period` — the phase-independent dedup identity.
        canonical: Vec<u8>,
    },
}

impl<'a> Segment<'a> {
    /// Start offset of this segment within the input.
    pub fn offset(&self) -> usize {
        match self {
            Segment::Solo(c) => c.offset(),
            Segment::Caterpillar { offset, .. } | Segment::Periodic { offset, .. } => *offset,
        }
    }

    /// Number of underlying chunks represented (1 for [`Segment::Solo`]).
    pub fn chunk_count(&self) -> usize {
        match self {
            Segment::Solo(_) => 1,
            Segment::Caterpillar { count, .. } => *count,
            Segment::Periodic { chunks, .. } => *chunks,
        }
    }

    /// Total number of bytes covered by this segment.
    pub fn len(&self) -> usize {
        match self {
            Segment::Solo(c) => c.len(),
            Segment::Caterpillar { unit, count, .. } => unit.len() * count,
            Segment::Periodic { total_len, .. } => *total_len,
        }
    }

    /// Whether this segment covers zero bytes (never true in practice).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The unique content to fingerprint/store for content addressing: the chunk
    /// bytes for [`Segment::Solo`], the repeated unit for [`Segment::Caterpillar`],
    /// or the canonical (phase-independent) period for [`Segment::Periodic`].
    /// Hash this; store it once; use [`offset`](Self::offset),
    /// [`len`](Self::len), and [`chunk_count`](Self::chunk_count) for the record.
    pub fn dedup_key(&self) -> &[u8] {
        match self {
            Segment::Solo(c) => &**c,
            Segment::Caterpillar { unit, .. } => unit,
            Segment::Periodic { canonical, .. } => canonical,
        }
    }
}

/// Wraps a [`SliceChunker`] and coalesces periodic / repeated regions.
pub struct CaterpillarChunker<'a, C> {
    data: &'a [u8],
    inner: SliceChunker<'a, C>,
    carry: Option<Chunk<'a>>,
    enable_period: bool,
    period_budget: usize,
}

impl<'a, C: Cdc> CaterpillarChunker<'a, C> {
    /// Creates a caterpillar chunker (run-length-encodes byte-identical adjacent
    /// chunks). This is the recommended default: free on data with no runs, and
    /// it collapses zero-fill, padding, and repeated blocks into single records.
    pub fn new(bytes: &'a [u8], min_size: usize, max_size: usize, cdc: C) -> Self {
        Self {
            data: bytes,
            inner: SliceChunker::new(bytes, min_size, max_size, cdc),
            carry: None,
            enable_period: false,
            period_budget: 0,
        }
    }

    /// EXPERIMENTAL — additionally enable period detection, which catches
    /// *phase-rotating* periodic runs that the default cannot (consecutive chunks
    /// that are rotations of the same period).
    ///
    /// This is usually **not** worth enabling: it adds a large per-chunk cost,
    /// and because mincdc self-aligns chunk boundaries to periods, the default
    /// already handles most periodic data. It only helps with narrow
    /// `[min, max]` windows where no period multiple fits. `budget` bounds total
    /// detection work in bytes (use `usize::MAX` for unlimited). See
    /// `examples/CATBENCH_RESULTS.md` for measurements.
    pub fn with_period_detection(mut self, budget: usize) -> Self {
        self.enable_period = true;
        self.period_budget = budget;
        self
    }

    /// Tier-2 gate + detector. Returns the smallest detected period `P` of the
    /// chunk `[start, end)`, or `None`. Charges the byte budget.
    fn detect_period(&mut self, start: usize, end: usize) -> Option<usize> {
        let len = end - start;
        if len < 2 * PROBE || self.period_budget < len {
            return None;
        }
        self.period_budget -= len;

        let data = self.data;
        let prefix = &data[start..start + PROBE];
        let max_p = (len / 2).min(MAX_PERIOD);

        let mut p = 1;
        while p <= max_p {
            // Cheap first-byte gate, then full prefix compare on a hit.
            if data[start + p] == prefix[0] && &data[start + p..start + p + PROBE] == prefix {
                // Verify the whole chunk is p-periodic: data[i] == data[i - p].
                if data[start + p..end]
                    .iter()
                    .zip(&data[start..end - p])
                    .all(|(a, b)| a == b)
                {
                    return Some(p);
                }
            }
            p += 1;
        }
        None
    }

    /// Does `[cstart, cend)` continue a `P`-periodic run (already periodic before
    /// `cstart`)? Checks `data[i] == data[i - P]`.
    fn continues_period(&self, cstart: usize, cend: usize, p: usize) -> bool {
        if cstart < p {
            return false;
        }
        let data = self.data;
        data[cstart..cend]
            .iter()
            .zip(&data[cstart - p..cend - p])
            .all(|(a, b)| a == b)
    }
}

impl<'a, C: Cdc> Iterator for CaterpillarChunker<'a, C> {
    type Item = Segment<'a>;

    fn next(&mut self) -> Option<Segment<'a>> {
        let first = self.carry.take().or_else(|| self.inner.next())?;
        let start = first.offset();
        let first_len = first.len();
        let unit: &'a [u8] = &self.data[start..start + first_len];

        // Tier 1: coalesce byte-identical adjacent chunks.
        let mut count = 1usize;
        let pending: Option<Chunk<'a>> = loop {
            match self.inner.next() {
                Some(c) if &*c == unit => count += 1,
                other => break other,
            }
        };
        if count >= 2 {
            self.carry = pending;
            return Some(Segment::Caterpillar { offset: start, unit, count });
        }

        // Tier 2: gated period detection on the single chunk `first`.
        if self.enable_period {
            if let Some(p) = self.detect_period(start, start + first_len) {
                let mut run_end = start + first_len;
                let mut chunks = 1usize;
                let mut nextc = pending;
                loop {
                    match nextc {
                        Some(c) if self.continues_period(c.offset(), c.offset() + c.len(), p) => {
                            run_end = c.offset() + c.len();
                            chunks += 1;
                            nextc = self.inner.next();
                        }
                        other => {
                            self.carry = other;
                            break;
                        }
                    }
                }
                if chunks >= 2 {
                    let raw_period = &self.data[start..start + p];
                    let canonical = canonical_rotation(raw_period);
                    return Some(Segment::Periodic {
                        offset: start,
                        raw_period,
                        total_len: run_end - start,
                        chunks,
                        canonical,
                    });
                }
                // Detected a period but it did not extend: not worth a record.
                // `self.carry` was already set in the loop.
                return Some(Segment::Solo(first));
            }
        }

        self.carry = pending;
        Some(Segment::Solo(first))
    }
}

/// Returns the least rotation of `s` (Booth's algorithm, O(n)). Used to give a
/// periodic cell a phase-independent dedup identity.
fn canonical_rotation(s: &[u8]) -> Vec<u8> {
    let k = least_rotation(s);
    let mut out = Vec::with_capacity(s.len());
    out.extend_from_slice(&s[k..]);
    out.extend_from_slice(&s[..k]);
    out
}

/// Booth's algorithm: index of the lexicographically least rotation of `s`.
fn least_rotation(s: &[u8]) -> usize {
    let n = s.len();
    if n == 0 {
        return 0;
    }
    let idx = |x: isize| -> u8 { s[x.rem_euclid(n as isize) as usize] };
    let mut f = vec![-1isize; 2 * n];
    let mut k: isize = 0;
    for j in 1..(2 * n) as isize {
        let sj = idx(j);
        let mut i = f[(j - k - 1) as usize];
        while i != -1 && sj != idx(k + i + 1) {
            if sj < idx(k + i + 1) {
                k = j - i - 1;
            }
            i = f[i as usize];
        }
        if sj != idx(k + i + 1) {
            // i == -1 here
            if sj < idx(k + i + 1) {
                k = j;
            }
            f[(j - k) as usize] = -1;
        } else {
            f[(j - k) as usize] = i + 1;
        }
    }
    k as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MinCdcHash4;

    const MIN: usize = 2 * 1024;
    const MAX: usize = 14 * 1024;

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

    /// Drives a chunker, asserts exact reconstruction + same underlying chunk
    /// count, and returns (plain_chunks, caterpillar_records).
    fn measure(label: &str, data: &[u8], full: bool) -> (usize, usize) {
        measure_mm(label, data, full, MIN, MAX)
    }

    fn measure_mm(label: &str, data: &[u8], full: bool, min: usize, max: usize) -> (usize, usize) {
        let cdc = MinCdcHash4::new();
        let plain = SliceChunker::new(data, min, max, cdc).count();

        let chunker = if full {
            CaterpillarChunker::new(data, min, max, cdc).with_period_detection(usize::MAX)
        } else {
            CaterpillarChunker::new(data, min, max, cdc)
        };

        let mut records = 0usize;
        let mut expanded = 0usize;
        let mut next_off = 0usize;
        let mut rebuilt: Vec<u8> = Vec::with_capacity(data.len());
        for s in chunker {
            assert_eq!(s.offset(), next_off, "{label}: offset not contiguous");
            records += 1;
            expanded += s.chunk_count();
            match &s {
                Segment::Solo(c) => rebuilt.extend_from_slice(c),
                Segment::Caterpillar { unit, count, .. } => {
                    for _ in 0..*count {
                        rebuilt.extend_from_slice(unit);
                    }
                }
                Segment::Periodic { raw_period, total_len, .. } => {
                    let mut written = 0;
                    while written < *total_len {
                        let take = raw_period.len().min(*total_len - written);
                        rebuilt.extend_from_slice(&raw_period[..take]);
                        written += take;
                    }
                }
            }
            next_off += s.len();
        }
        assert_eq!(rebuilt, data, "{label}: must reconstruct input exactly");
        assert_eq!(expanded, plain, "{label}: must represent same chunk count");

        let pct = if plain > 0 {
            100.0 * (1.0 - records as f64 / plain as f64)
        } else {
            0.0
        };
        let tier = if full { "full " } else { "simple" };
        println!("{label:<24} {tier}  plain={plain:>5}  records={records:>5}  -{pct:>5.1}%");
        (plain, records)
    }

    #[test]
    fn booth_least_rotation_is_correct() {
        // Brute-force oracle for small inputs.
        fn brute(s: &[u8]) -> Vec<u8> {
            (0..s.len())
                .map(|k| {
                    let mut r = s[k..].to_vec();
                    r.extend_from_slice(&s[..k]);
                    r
                })
                .min()
                .unwrap_or_default()
        }
        for seed in 0..200u64 {
            let n = 1 + (seed as usize % 17);
            let s = xorshift(seed + 1, n);
            assert_eq!(canonical_rotation(&s), brute(&s), "seed {seed}");
        }
        assert_eq!(canonical_rotation(b"bca"), b"abc");
        assert_eq!(canonical_rotation(b"abcabc"), b"abcabc");
    }

    #[test]
    fn tier1_already_handles_period_aligned_data() {
        // FINDING: mincdc self-aligns boundaries to periods when a period multiple
        // fits in [min, max] (here chunk_len ~= 3 * 777), so consecutive chunks are
        // byte-identical and tier 1 alone collapses the region. Tier 2 adds little.
        let period = xorshift(42, 777);
        let mut data = Vec::new();
        while data.len() < 4 * 1024 * 1024 {
            data.extend_from_slice(&period);
        }
        let (plain, simple) = measure("period777 wide [2k,14k]", &data, false);
        let (_, full) = measure("period777 wide [2k,14k]", &data, true);
        assert!(simple * 20 < plain, "tier 1 already collapses aligned periodic data");
        assert!(full <= simple, "tier 2 must never be worse than tier 1");
    }

    #[test]
    fn tier2_wins_when_no_period_multiple_fits() {
        // Force genuine rotation: with [2048, 2200] there is NO multiple of 777 in
        // range, so mincdc cannot align -> every chunk is a different rotation ->
        // tier 1 fails. Tier 2's period detection still collapses it.
        let period = xorshift(42, 777);
        let mut data = Vec::new();
        while data.len() < 4 * 1024 * 1024 {
            data.extend_from_slice(&period);
        }
        let (plain, simple) = measure_mm("period777 narrow [2k,2.2k]", &data, false, 2048, 2200);
        let (_, full) = measure_mm("period777 narrow [2k,2.2k]", &data, true, 2048, 2200);

        assert!(simple as f64 > plain as f64 * 0.9, "tier 1 cannot coalesce rotations");
        assert!(full * 20 < plain, "tier 2 collapses the rotating run");
    }

    #[test]
    fn full_still_wins_on_low_entropy_and_is_noop_on_random() {
        // Zero-fill: tier 1 already nails it; tier 2 must not regress it.
        let (plain, records) = measure("zero-fill 1MiB", &vec![0u8; 1024 * 1024], true);
        assert!(records * 10 < plain);

        // Random: no coalescing in either tier; layer is ~free and lossless.
        let data = xorshift(1234, 1024 * 1024);
        let (plain, records) = measure("random 1MiB", &data, true);
        assert!(
            records as f64 >= plain as f64 * 0.98,
            "must not coalesce random data"
        );
    }

    #[test]
    fn budget_floors_detection_work() {
        // With a zero budget, the full chunker degrades to tier 1 only.
        let period = xorshift(7, 999);
        let mut data = Vec::new();
        while data.len() < 1024 * 1024 {
            data.extend_from_slice(&period);
        }
        let cdc = MinCdcHash4::new();
        let plain = SliceChunker::new(&data, MIN, MAX, cdc).count();

        // Narrow window so tier 1 cannot align: tier 2 is the only thing that can
        // collapse it, and only when the budget allows detection.
        let (min, max) = (2048usize, 2200usize);
        let plain_narrow = SliceChunker::new(&data, min, max, cdc).count();
        let starved = CaterpillarChunker::new(&data, min, max, cdc)
            .with_period_detection(0)
            .count();
        let fed = CaterpillarChunker::new(&data, min, max, cdc)
            .with_period_detection(usize::MAX)
            .count();

        assert!(
            starved as f64 > plain_narrow as f64 * 0.9,
            "budget 0 disables period detection"
        );
        assert!(fed * 20 < starved, "budget enables the collapse");
        println!("budget: plain={plain} narrow_plain={plain_narrow} starved={starved} fed={fed}");
    }
}
