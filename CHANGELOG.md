# Changelog

## 0.6.0 — 2026-07-10

Packed-scanning caterpillar fast path (VectorCDC-style SIMD).

- Inside a repetitive run, the caterpillar no longer pays a full argmin
  boundary search plus a memcmp per chunk. One packed equality scan
  (`byte_run_len` broadcast compare for constant bytes, `common_prefix_len`
  self-shifted compare for longer periods) proves the run periodic, and every
  chunk whose decision window stays inside the periodic region is emitted
  directly — the boundary search provably returns the same split, so it is
  skipped. See `packed_repeats` in `src/caterpillar.rs` for the proof.
- New SIMD primitives in all backends: NEON (`vceqq_u8` + `vshrn` mask),
  x86_64 (SSE2 baseline, AVX2, AVX-512BW via runtime dispatch), and a
  word-at-a-time scalar fallback.
- Applies to both `CaterpillarChunker` and `CaterpillarReadChunker`. Output is
  bit-identical to 0.5.0 (same segments, same grouping, same bytes).
- Criterion bench (`cargo bench`, min=2048 max=14336), caterpillar 0.5 → 0.6:
  - Apple M-series (NEON): zeros 1.78 → 30.2 GiB/s, periodic-777 2.03 → 27.5
    GiB/s, random+zero-hole 2.96 → 13.3 GiB/s, random unchanged (~8.3 GiB/s).
  - Intel Xeon w/ AVX-512BW (dedicated Fly performance-4x): zeros 2.41 → 74.3
    GiB/s, periodic-777 2.54 → 22.9 GiB/s, random+zero-hole 3.51 → 13.8 GiB/s,
    random unchanged (~10.2 GiB/s).

  The caterpillar previously cost up to ~8% on redundant data; it is now
  9–30x faster than plain mincdc there.
- Tests: a `packed_repeats` soundness test against the real boundary search as
  oracle (break position swept over every byte at every alignment), a
  segment-stream differential against the pre-SIMD caterpillar (adversarial
  corpus + proptest), per-width SIMD agreement tests, and a streaming corpus
  entry with a period break inside the final decision window.

## 0.5.0 and earlier

See git history (`git log --oneline`).
