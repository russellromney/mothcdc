//! End-to-end dedup round-trip from the user's perspective.
//!
//! This models what a real dedup store does, exactly as the README documents:
//! chunk -> store each segment's `dedup_key()` under its hash -> later, restore
//! purely from (store + manifest) by tiling the stored bytes to `len` -> assert
//! the bytes come back identical.
//!
//! This is the test that was missing: the in-crate unit tests reconstructed from
//! the segment's own fields, never from the *stored* `dedup_key()`, so they could
//! not catch a `dedup_key()` that isn't reconstruction-safe (the period-detection
//! variant once returned a rotated key and silently corrupted off-phase runs).

use std::collections::HashMap;
use std::io::Cursor;

use mincatcdc::{CaterpillarChunker, CaterpillarReadChunker, MinCdcHash4, SliceChunker};
use proptest::prelude::*;

fn fnv1a(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
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

/// `period`: `None` = default (tier-1 only); `Some(budget)` = enable period
/// detection with that byte budget.
fn build(
    data: &[u8],
    min: usize,
    max: usize,
    period: Option<usize>,
) -> CaterpillarChunker<'_, MinCdcHash4> {
    let c = CaterpillarChunker::new(data, min, max, MinCdcHash4::new());
    match period {
        Some(budget) => c.with_period_detection(budget),
        None => c,
    }
}

/// The full user round-trip: chunk, build a content-addressed store keyed by
/// `hash(dedup_key())` plus an ordered manifest, then restore from store +
/// manifest only and assert it equals the input. Also checks `reconstruct_into`.
fn assert_roundtrip(label: &str, data: &[u8], min: usize, max: usize, period: Option<usize>) {
    let tag = format!("{label} (min={min} max={max} period={period:?})");

    // --- ingest ---
    let mut store: HashMap<u64, Vec<u8>> = HashMap::new();
    let mut manifest: Vec<(u64, usize)> = Vec::new(); // (key_hash, logical_len)
    let mut next_off = 0usize;
    for seg in build(data, min, max, period) {
        assert_eq!(seg.offset(), next_off, "{tag}: non-contiguous offset");
        assert!(!seg.is_empty(), "{tag}: empty segment");
        // Note: a coalesced segment's len() intentionally exceeds max (it stands
        // for a whole run of chunks), so max only bounds the underlying chunks.
        let key = seg.dedup_key();
        let h = fnv1a(key);
        store.entry(h).or_insert_with(|| key.to_vec());
        manifest.push((h, seg.len()));
        next_off += seg.len();
    }
    assert_eq!(next_off, data.len(), "{tag}: coverage gap");

    // The caterpillar must represent exactly the same underlying chunks as plain
    // mincdc — it only groups them, never adds or drops any.
    let plain = SliceChunker::new(data, min, max, MinCdcHash4::new()).count();
    let expanded: usize = build(data, min, max, period).map(|s| s.chunk_count()).sum();
    assert_eq!(
        expanded, plain,
        "{tag}: chunk_count {expanded} != plain {plain}"
    );

    // --- restore from store + manifest only (the documented path) ---
    let mut restored = Vec::with_capacity(data.len());
    for (h, len) in &manifest {
        let bytes = store
            .get(h)
            .expect("manifest references a chunk not in the store");
        let mut w = 0;
        while w < *len {
            let take = bytes.len().min(*len - w);
            restored.extend_from_slice(&bytes[..take]);
            w += take;
        }
    }
    assert_eq!(restored, data, "{tag}: store round-trip corrupted the data");

    // --- reconstruct_into helper must agree too ---
    let mut via_helper = Vec::with_capacity(data.len());
    for seg in build(data, min, max, period) {
        seg.reconstruct_into(&mut via_helper);
    }
    assert_eq!(via_helper, data, "{tag}: reconstruct_into mismatch");
}

fn corpora() -> Vec<(&'static str, Vec<u8>)> {
    let mut periodic = Vec::new();
    let p = xorshift(9, 777);
    while periodic.len() < 1024 * 1024 {
        periodic.extend_from_slice(&p);
    }
    let mut holed = xorshift(2, 1024 * 1024);
    for b in holed[256 * 1024..768 * 1024].iter_mut() {
        *b = 0;
    }
    vec![
        ("random", xorshift(1, 1024 * 1024)),
        ("zeros", vec![0u8; 1024 * 1024]),
        ("const-0xAB", vec![0xABu8; 1024 * 1024]),
        ("periodic-777", periodic),
        ("random+zero-hole", holed),
        ("tiny", xorshift(5, 100)),
        ("empty", Vec::new()),
    ]
}

#[test]
fn roundtrip_default_and_period_wide_and_narrow() {
    for (name, data) in corpora() {
        for period in [None, Some(usize::MAX)] {
            // Wide window (normal CDC) and a narrow window (forces real rotation
            // when period detection is on).
            assert_roundtrip(name, &data, 2048, 14336, period);
            assert_roundtrip(name, &data, 2048, 2200, period);
        }
    }
}

#[test]
fn roundtrip_offphase_periodic_with_detection() {
    // The exact shape that the old canonical-rotation dedup_key corrupted: an
    // aperiodic prefix pushes the periodic run onto a non-minimal phase, and a
    // narrow window prevents mincdc from aligning, so tier-2 period detection
    // fires off-phase. Must still round-trip losslessly.
    let mut data = xorshift(123, 2500); // aperiodic prefix
    let period = xorshift(7, 100);
    while data.len() < 4 * 1024 * 1024 {
        data.extend_from_slice(&period);
    }
    assert_roundtrip("offphase-period", &data, 2048, 2200, Some(usize::MAX));
}

/// A reader that returns at most `step` bytes per call, to exercise the streaming
/// chunker's buffer refill/shift logic (and its reader-dependent run splitting).
struct ChokedReader<'a> {
    data: &'a [u8],
    pos: usize,
    step: usize,
}
impl std::io::Read for ChokedReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.step.min(buf.len()).min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Drives the streaming caterpillar over `reader`, asserting contiguity, lossless
/// reconstruction, exact coverage, and that it represents the same underlying
/// chunk count as plain mincdc. (It is NOT asserted to match the slice
/// caterpillar's segment grouping — the streaming version splits long runs at
/// buffer boundaries, which depends on the reader.)
fn assert_stream_roundtrip<R: std::io::Read>(
    tag: &str,
    reader: R,
    data: &[u8],
    min: usize,
    max: usize,
) {
    let cdc = MinCdcHash4::new();
    let plain = SliceChunker::new(data, min, max, cdc).count();
    let mut rc = CaterpillarReadChunker::new(reader, min, max, cdc);
    let (mut next_off, mut chunks) = (0usize, 0usize);
    let mut rebuilt = Vec::with_capacity(data.len());
    while let Some(s) = rc.next().unwrap() {
        assert_eq!(s.offset(), next_off, "{tag}: non-contiguous");
        assert!(!s.is_empty(), "{tag}: empty segment");
        chunks += s.chunk_count();
        next_off += s.len();
        s.reconstruct_into(&mut rebuilt);
    }
    assert_eq!(rebuilt, data, "{tag}: stream round-trip corrupted data");
    assert_eq!(next_off, data.len(), "{tag}: coverage gap");
    assert_eq!(
        chunks, plain,
        "{tag}: chunk_count {chunks} != plain {plain}"
    );
}

#[test]
fn streaming_caterpillar_roundtrips() {
    for (name, data) in corpora() {
        for (min, max) in [(2048usize, 14336usize), (2048, 2200), (64, 256)] {
            // One big read (Cursor) and a hostile 1-byte reader — the run splits
            // land differently, but both must be lossless and fully cover.
            assert_stream_roundtrip(
                &format!("{name}/cursor"),
                Cursor::new(&data),
                &data,
                min,
                max,
            );
            let choked = ChokedReader {
                data: &data,
                pos: 0,
                step: 1,
            };
            assert_stream_roundtrip(&format!("{name}/choked"), choked, &data, min, max);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, ..ProptestConfig::default() })]

    /// Fuzz the full round-trip over random data, sizes, and budgets — including
    /// partial budgets and period detection on/off. Shrinks any state-machine
    /// or reconstruction bug to a minimal case.
    #[test]
    fn prop_caterpillar_roundtrip(
        data in proptest::collection::vec(any::<u8>(), 0..8192),
        min in 16usize..512,
        extra in 0usize..512,
        period in proptest::option::of(0usize..100_000),
    ) {
        assert_roundtrip("prop", &data, min, min + extra, period);
    }

    /// Fuzz the streaming caterpillar over random data, sizes, and reader chunk
    /// sizes — asserts lossless reconstruction, full coverage, and same chunk
    /// count as plain mincdc regardless of how the reader fragments input.
    #[test]
    fn prop_streaming_caterpillar_roundtrip(
        data in proptest::collection::vec(any::<u8>(), 0..8192),
        min in 16usize..512,
        extra in 0usize..512,
        step in 1usize..4096,
    ) {
        let choked = ChokedReader { data: &data, pos: 0, step };
        assert_stream_roundtrip("prop-stream", choked, &data, min, min + extra);
    }
}
