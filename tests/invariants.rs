//! Behavioral test harness for MinCDC.
//!
//! Rather than trusting that the SIMD implementation is correct, this suite
//! checks the chunkers against a set of *invariants* and against an independent
//! brute-force reference (an "oracle"). It exercises only the public API, so it
//! is decoupled from the internals (scalar/NEON/x86 SIMD) and verifies whatever
//! implementation the current target compiled in.
//!
//! Tiers:
//!  1. Structural invariants  -- output is a well-formed partition (Tier 1).
//!  2. Differential / oracle  -- output equals a naive spec implementation.
//!  3. Property-based + corpus -- random (proptest, shrinking) + adversarial.
//!  4. Edit-locality          -- the property that makes CDC useful for dedup.
//!  5. Reader/Slice agreement -- ReadChunker == SliceChunker under a hostile
//!     (1-byte) reader that stresses buffer refills.

#![allow(deprecated)] // MinCdc4 is deprecated but we still want to test it.

use std::io::{self, Read};

use mothcdc::mincdc::{Cdc, MinCdc4, MinCdcHash4, ReadChunker, SliceChunker};
use proptest::prelude::*;

const WINDOW: usize = 4;

// Fixed hash parameters for the oracle to mirror (multiplier must be odd).
const HASH_M: u32 = 0x915f_77f5;
const HASH_A: u32 = 0x3463_6463;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Plain,  // MinCdc4: evaluate(window) = u32::from_le_bytes(window)
    Hashed, // MinCdcHash4: evaluate(window) = hash(u32::from_le_bytes(window))
}

fn hash_params(mode: Mode) -> Option<(u32, u32)> {
    match mode {
        Mode::Plain => None,
        Mode::Hashed => Some((HASH_M, HASH_A)),
    }
}

// ---------------------------------------------------------------------------
// Independent oracle: the documented algorithm, written as plainly as possible.
// No SIMD, no clever tail handling -- this is what we differentially test
// against. It is deliberately derived from the README/lib.rs prose, not from
// the optimized code, so a shared bug is unlikely.
// ---------------------------------------------------------------------------

fn evaluate(window: &[u8], hash: Option<(u32, u32)>) -> u32 {
    let v = u32::from_le_bytes(window.try_into().unwrap());
    match hash {
        Some((m, a)) => v.wrapping_mul(m).wrapping_add(a),
        None => v,
    }
}

/// Returns the chunk boundaries as `(offset, len)` pairs.
fn oracle_chunks(
    bytes: &[u8],
    min: usize,
    max: usize,
    hash: Option<(u32, u32)>,
) -> Vec<(usize, usize)> {
    assert!(min <= max && max > 0);
    let mut chunks = Vec::new();
    let mut s = 0usize;
    while s < bytes.len() {
        let left = bytes.len() - s;

        // The final chunk may be shorter than min_size.
        if left <= min {
            chunks.push((s, left));
            break;
        }

        // Search window for the splitpoint: chunk length i in [min, max], where
        // the evaluated window is bytes[s+i-WINDOW .. s+i]. Mirrors the slice
        // arithmetic in lib.rs (start saturates when min < WINDOW).
        let start = s + min.saturating_sub(WINDOW);
        let stop = (s + max).min(bytes.len());
        let search = &bytes[start..stop];

        let ideal = if search.len() < WINDOW {
            search.len()
        } else {
            // Earliest window minimizing the evaluation (ties -> earliest).
            let mut best_i = 0usize;
            let mut best_v = u32::MAX;
            for i in 0..=search.len() - WINDOW {
                let v = evaluate(&search[i..i + WINDOW], hash);
                if v < best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            WINDOW + best_i
        };

        let split = start + ideal;
        debug_assert!(split > s && split <= bytes.len());
        chunks.push((s, split - s));
        s = split;
    }
    chunks
}

// ---------------------------------------------------------------------------
// Collecting chunks from the real chunkers.
// ---------------------------------------------------------------------------

fn slice_chunks(data: &[u8], min: usize, max: usize, mode: Mode) -> Vec<(usize, usize)> {
    fn collect<C: Cdc>(data: &[u8], min: usize, max: usize, cdc: C) -> Vec<(usize, usize)> {
        SliceChunker::new(data, min, max, cdc)
            .map(|c| (c.offset(), c.len()))
            .collect()
    }
    match mode {
        Mode::Plain => collect(data, min, max, MinCdc4::new()),
        Mode::Hashed => collect(data, min, max, MinCdcHash4::with_params(HASH_M, HASH_A)),
    }
}

/// A `Read` impl that yields at most `step` bytes per call, to stress the
/// buffering/refill logic in `ReadChunker` (which `Cursor` never exercises).
struct ChokedReader<'a> {
    data: &'a [u8],
    pos: usize,
    step: usize,
}

impl<'a> Read for ChokedReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.step.min(buf.len()).min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

fn read_chunks(
    data: &[u8],
    min: usize,
    max: usize,
    mode: Mode,
    step: usize,
) -> Vec<(usize, usize)> {
    fn collect<C: Cdc>(
        data: &[u8],
        min: usize,
        max: usize,
        cdc: C,
        step: usize,
    ) -> Vec<(usize, usize)> {
        let reader = ChokedReader { data, pos: 0, step };
        let mut chunker = ReadChunker::new(reader, min, max, cdc);
        let mut out = Vec::new();
        while let Some(chunk) = chunker.next().unwrap() {
            out.push((chunk.offset(), chunk.len()));
        }
        out
    }
    match mode {
        Mode::Plain => collect(data, min, max, MinCdc4::new(), step),
        Mode::Hashed => collect(
            data,
            min,
            max,
            MinCdcHash4::with_params(HASH_M, HASH_A),
            step,
        ),
    }
}

// ---------------------------------------------------------------------------
// Tier 1: structural invariants on the output of any chunker.
// ---------------------------------------------------------------------------

fn assert_structural(chunks: &[(usize, usize)], input_len: usize, min: usize, max: usize) {
    if input_len == 0 {
        assert!(chunks.is_empty(), "empty input must yield no chunks");
        return;
    }

    let mut expected_offset = 0usize;
    for (i, &(off, len)) in chunks.iter().enumerate() {
        // Contiguous, gap-free, in order.
        assert_eq!(off, expected_offset, "chunk {i} offset not contiguous");
        // No empty chunks.
        assert!(len > 0, "chunk {i} is empty");
        // Maximum size is *always* respected.
        assert!(len <= max, "chunk {i} len {len} exceeds max {max}");
        // Every chunk but the last respects the minimum (min >= WINDOW assumed).
        if i + 1 < chunks.len() {
            assert!(len >= min, "non-final chunk {i} len {len} below min {min}");
        }
        expected_offset += len;
    }
    // Full coverage: chunks tile the entire input.
    assert_eq!(expected_offset, input_len, "chunks do not cover input");
}

/// Run every applicable invariant + the oracle differential for one config.
fn check_all(data: &[u8], min: usize, max: usize, mode: Mode) {
    let got = slice_chunks(data, min, max, mode);

    // Tier 1.
    assert_structural(&got, data.len(), min, max);

    // Determinism: chunking is a pure function of (data, min, max, params).
    let again = slice_chunks(data, min, max, mode);
    assert_eq!(got, again, "chunking is not deterministic");

    // Tier 2: differential vs the independent oracle.
    let want = oracle_chunks(data, min, max, hash_params(mode));
    assert_eq!(
        got,
        want,
        "slice chunker disagrees with oracle (mode={mode:?}, min={min}, max={max}, len={})",
        data.len()
    );

    // Tier 5: ReadChunker must match SliceChunker, even with a choked reader.
    for &step in &[1usize, 3, 7, 64, 4096] {
        let read = read_chunks(data, min, max, mode, step);
        assert_eq!(
            read, got,
            "read chunker (step={step}) disagrees with slice chunker (mode={mode:?})"
        );
    }
}

// ---------------------------------------------------------------------------
// Tier 3a: adversarial corpus (deterministic, hand-picked nasty inputs).
// ---------------------------------------------------------------------------

fn corpus() -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = Vec::new();
    // Degenerate sizes.
    for n in [
        0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 1000,
    ] {
        v.push(vec![0u8; n]); // all zeros (global min for Plain mode)
        v.push(vec![0xFFu8; n]); // all 0xFF (global max)
        v.push((0..n).map(|i| i as u8).collect()); // counter
    }
    // Periodic patterns with periods near WINDOW / typical sizes.
    for period in [1usize, 2, 3, 4, 5, 6, 7, 8, 16, 31, 32] {
        let pat: Vec<u8> = (0..2048).map(|i| (i % period) as u8).collect();
        v.push(pat);
    }
    // A long zero run embedded in random data (ties everywhere in Plain mode).
    {
        let mut seed = 0xC0FFEEu32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 24) as u8
        };
        let mut buf: Vec<u8> = (0..4096).map(|_| rng()).collect();
        for b in buf.iter_mut().take(2048).skip(1024) {
            *b = 0;
        }
        v.push(buf);
    }
    v
}

#[test]
fn corpus_all_invariants() {
    let sizes = [
        (4usize, 4usize),
        (4, 8),
        (8, 16),
        (16, 64),
        (32, 32),
        (50, 200),
    ];
    for data in corpus() {
        for &(min, max) in &sizes {
            check_all(&data, min, max, Mode::Plain);
            check_all(&data, min, max, Mode::Hashed);
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 3b: large random buffers to exercise the SIMD main loops + tail paths.
// ---------------------------------------------------------------------------

fn xorshift_bytes(seed: u64, n: usize) -> Vec<u8> {
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

#[test]
fn large_random_differential() {
    // Large min/max so each search window is big enough to drive the widest
    // SIMD paths (e.g. AVX-512 needs windows > 64 bytes).
    let configs = [
        (4usize, 4usize),
        (64, 256),
        (256, 1024),
        (1024, 4096),
        (4096, 4097),
    ];
    for seed in 0..8u64 {
        let data = xorshift_bytes(seed.wrapping_add(1), 256 * 1024);
        for &(min, max) in &configs {
            check_all(&data, min, max, Mode::Plain);
            check_all(&data, min, max, Mode::Hashed);
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 3c: property-based testing with shrinking.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn prop_invariants_and_oracle(
        data in proptest::collection::vec(any::<u8>(), 0..8192),
        min in 4usize..400,
        extra in 0usize..400,
    ) {
        let max = min + extra;
        check_all(&data, min, max, Mode::Plain);
        check_all(&data, min, max, Mode::Hashed);
    }
}

// ---------------------------------------------------------------------------
// Tier 4: edit-locality -- the property that makes CDC valuable for dedup.
// ---------------------------------------------------------------------------

/// Insert `blob` into `data` at position `p`.
fn insert_at(data: &[u8], p: usize, blob: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + blob.len());
    out.extend_from_slice(&data[..p]);
    out.extend_from_slice(blob);
    out.extend_from_slice(&data[p..]);
    out
}

/// HARD invariant: any chunk whose entire decision window lies before the edit
/// (offset + max <= p) is preserved byte-for-byte. This is provable: such a
/// chunk's boundary is decided solely by bytes < p, which the insertion does
/// not touch. If this ever fails, deduplication of unchanged prefixes is broken.
#[test]
fn locality_prefix_is_preserved() {
    for mode in [Mode::Plain, Mode::Hashed] {
        for seed in 0..16u64 {
            let data = xorshift_bytes(seed + 100, 16384);
            let (min, max) = (64usize, 256usize);
            let p = ((seed as usize) * 997) % data.len();
            let blob = xorshift_bytes(seed + 555, 1 + (seed as usize % 200));
            let edited = insert_at(&data, p, &blob);

            let base = slice_chunks(&data, min, max, mode);
            let after = slice_chunks(&edited, min, max, mode);
            use std::collections::HashSet;
            let after_starts: HashSet<usize> = after.iter().map(|&(o, _)| o).collect();

            for &(off, len) in &base {
                if off + max <= p {
                    assert!(
                        after_starts.contains(&off),
                        "boundary at {off} lost after edit at {p} (mode={mode:?})"
                    );
                    // And the chunk content is identical (same len at same offset).
                    let same_len = after.iter().any(|&(o, l)| o == off && l == len);
                    assert!(same_len, "chunk at {off} changed despite being before edit");
                }
            }
        }
    }
}

/// SOFT metric (aggregate, not per-case): after an edit, how much of the tail
/// resynchronizes? With CDC the boundaries past the edit should realign (shifted
/// by the insert length). We assert only an aggregate floor to avoid flakiness,
/// while still catching a regression that destroys resync entirely.
#[test]
fn locality_resync_metric() {
    use std::collections::HashSet;
    let mode = Mode::Hashed; // the recommended, well-behaved default
    let (min, max) = (32usize, 128usize);

    let mut total_tail = 0usize;
    let mut matched_tail = 0usize;
    let trials = 200u64;

    for seed in 0..trials {
        let data = xorshift_bytes(seed + 7, 32768);
        let p = ((seed as usize) * 1234577) % data.len();
        let k = 1 + (seed as usize % 97);
        let blob = xorshift_bytes(seed + 31, k);
        let edited = insert_at(&data, p, &blob);

        let base = slice_chunks(&data, min, max, mode);
        let after = slice_chunks(&edited, min, max, mode);
        let after_starts: HashSet<usize> = after.iter().map(|&(o, _)| o).collect();

        // Consider base boundaries well past the edit; their counterpart in the
        // edited stream should be at offset + k once resynchronized.
        for &(off, _) in &base {
            if off >= p + max {
                total_tail += 1;
                if after_starts.contains(&(off + k)) {
                    matched_tail += 1;
                }
            }
        }
    }

    let ratio = matched_tail as f64 / total_tail.max(1) as f64;
    println!("resync: {matched_tail}/{total_tail} tail boundaries realigned ({ratio:.3})");
    // For random data MinCDCHash4 resyncs almost completely; a floor of 0.7
    // catches a catastrophic regression without flaking on the resync gap.
    assert!(
        ratio > 0.7,
        "tail resynchronization collapsed: only {ratio:.3} of boundaries realigned"
    );
}
