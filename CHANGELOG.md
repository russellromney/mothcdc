# Changelog

## 0.7.1 — API unification and release hardening (2026-07-12)

Unifies the public API around the caterpillar layer as the recommended
entry point, and separates the inherited MinCDC core from the caterpillar
layer at the module level:

- `caterpillar::CaterpillarChunker` and `caterpillar::CaterpillarReadChunker`
  are renamed to `MothChunker` and `MothReadChunker` (still re-exported at
  the crate root). No deprecated aliases — the old names are gone.
- `MothChunker` and `MothReadChunker` are now generic over the `Cdc`
  implementation with `MinCdcHash4` as the default type parameter.
  `MothChunker::new(bytes, min, max)` / `MothReadChunker::new(reader, min,
  max)` build with `MinCdcHash4::new()` internally; `with_cdc(..., cdc)`
  takes a custom `Cdc` instance (e.g. `MinCdcHash4::with_params`, or the
  academic `MinCdc4`).
- The inherited MinCDC core (`Cdc`, `MinCdc4`, `MinCdcHash4`, `SliceChunker`,
  `ReadChunker`) moved into a new `mincdc` submodule and is no longer
  re-exported at the crate root — reach it as `mothcdc::mincdc::...`. Only
  `Chunk` stays re-exported at the root (shared by both layers).
- `Chunk`/`Segment` offsets, segment lengths, and represented chunk counts are
  now `u64`; `Segment` has a private validated representation rather than
  publicly constructible invalid variants.
- Reader chunkers retry `Interrupted`, use checked buffer arithmetic, expose
  fallible constructors and reader accessors, and report offset/count overflow.
- Streaming caterpillar runs now remain maximal even when a `Read`
  implementation returns tiny fragments; a one-byte reader and `Cursor`
  produce the same grouping.
- Fixed a panic on x86_64 CPUs without SSE4.1 when the exact scalar argmin was
  one of the final three windows.
- The public `Cdc` splitpoint contract is documented and enforced.
- The academic `MinCdc4` unit struct can be constructed directly without using
  its deprecated `new()` helper.
- The C API validates size configurations without panicking. A normal Rust
  dependency now builds only an `rlib`; the benchmark static library is built
  explicitly with `cargo rustc --features capi -- --crate-type staticlib`.
- CI now covers every feature, formatting, the 1.89 MSRV, and a 32-bit target.
- The SIMD prefetch soundness fix was submitted upstream as
  [orlp/mincdc#1](https://github.com/orlp/mincdc/pull/1).

Also includes the mincatcdc -> mothcdc rename (no functional change).
Credits unchanged: MinCDC algorithm (Orson Peters), caterpillar layer
inspired by Chonkers (Berger), vector acceleration in the style of
VectorCDC (Udayashankar et al.).

## 0.6.0 — optimization campaign and packed scanning (2026-07-10–11)

Ten hypothesis-driven loops on a fixed Fly performance-4x machine (AMD
EPYC/AVX2), each benched against the same corpus (public Tigris bucket).
Kept (4): loop 2 — 4x-unrolled packed-scan skip loops (zeros scan 46 → 79
GiB/s, +2-3% end-to-end on real VM images); loop 5 — the streaming
caterpillar carries runs across buffer refills (a multi-GB zero region is
now ONE record instead of one per 4 MiB buffer; unit copied once per
crossing run, everything else stays zero-copy); loop 6/7 — `examples/frontier.rs`
sweep tool plus measured configuration guidance: at the same ~8 KiB average,
`(4096, 12288)` chunks ~40% faster than `(2048, 14336)` with equal-or-better
dedup (in dedup-bench's own harness: 16.2 GiB/s and 53.29% savings on the
raw Debian image — past VectorCDC-AE-Min's 13.8 GiB/s on the same run);
loop 9 — the guidance in the README.
Rejected with data (4): fat LTO/codegen-units (±1%, kernels are monolithic
target_feature fns); dual-accumulator argmin (±2%, the OoO core already
extracts that ILP on both EPYC and M-series); prefetch-distance tuning
(±2% across 4K-64K); branch-skipping the argmin offset blend (-9% to -40%,
the branch mispredicts exactly where the argmin is hot — the unconditional
blend design is correct). Hash multiplier choice: measured irrelevant
(three multipliers within noise); `a=0` trades +1.4pp savings for chunk-size
skew and -15% speed — documented, not defaulted.

Backup-generation corpus (LNX: 6 consecutive linux-6.6.x source tars, 8 GB,
dedup-bench harness) — added because VM images are not the typical dedup
workload — found and fixed a real caterpillar regression: tars of similar
trees are full of medium pseudo-periodic stretches where the unbounded
extent scan ran long and yielded nothing (-31% vs plain mincdc). The scan
is now staged behind a one-unit probe (bounded like the pre-0.6 memcmp),
shrinking the overhead to ~5% on that corpus, with identical output.
LNX results: mincdc dedup 59.17% and the narrow (4096,12288) config 60.28%
— the best space savings of every algorithm measured (SeqCDC 56.45%,
AE-Min 55.47%, FastCDC 52.44%, RAM 49.21%) — at 8.3-9.9 GB/s vs
VectorCDC-AE-Min's 4.6 GB/s on the same data (only VectorCDC-RAM is
faster at 17.2 GB/s, with the worst dedup of the field).

### Packed-scanning caterpillar fast path

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
- Bench results (`cargo bench`, min=2048 max=14336), caterpillar 0.5 → 0.6.
  Real data first — synthetic pure-zeros is the ceiling, not the claim:
  - Raw Debian 12 VM image (uncompressed, first 768 MiB): 5.0 → 13.0 GiB/s
    (2.6x) on AMD EPYC/AVX2; 3.0 → 11.2 GiB/s (3.8x) on Apple M-series/NEON.
  - Mostly-empty 200 MiB disk image: 1.8 → 30.9 GiB/s (17x, NEON).
  - Zero-padded build artifact: 5.0 → 8.1 GiB/s (1.6x, NEON).
  - FAST'25 DEB dataset (.ova appliances — streamOptimized/compressed VMDKs,
    so no byte-identical runs exist): no change, ~12 GiB/s all variants on
    EPYC/AVX2. Compressed or run-free data (SQLite, logs, random) is a no-op
    within ~2%.
  - Synthetic ceiling: zeros 2.4 → 74.3 GiB/s on Intel Xeon/AVX-512BW,
    1.8 → 30.2 GiB/s on NEON.
- New `capi` feature: a minimal C API (`mothcdc_next_chunk`) plus a
  dedup-bench fork
  (github.com/russellromney/dedup-bench, branch `mincatcdc-integration`)
  that adds `chunking_algo=mincdc`, so mincatcdc is measured by the *same*
  harness, timers, and dedup measurement as every other chunker. A unit test
  drives the C API under dedup-bench's exact buffering protocol and asserts
  boundary equality with `SliceChunker`; hash outputs of plain and
  caterpillar modes are byte-identical by construction (verified on NEON and
  x64).
- Head-to-head in dedup-bench itself (one AMD EPYC/AVX2 machine, chunking
  only, ~8 KiB targets). Raw Debian VM image: mincdc 4.8 GiB/s,
  mincdc+caterpillar 10.1 GiB/s — vs AE-Min 1.3, FastCDC 1.8, SeqCDC 2.7,
  VectorCDC-AE-Max 9.3, VectorCDC-AE-Min 14.7, VectorCDC-RAM 24.3. DEB .ova
  subset (FAST'25 dataset, compressed VMDKs): mincdc 11.6 GiB/s — faster
  than every content-defined competitor except VectorCDC-RAM (21.3);
  caterpillar neutral (-0.5%).
- The metrics that matter beyond speed, same runs: **space savings** — mincdc
  has the best dedup ratio of all algorithms measured on both datasets
  (raw image 53.19% vs FastCDC 51.77% / AE-Min 50.02%; DEB subset 8.14% vs
  FastCDC 7.88% / AE-Min 6.16%), and the caterpillar preserves it exactly.
  **Metadata records** — on the raw image the caterpillar collapses 231,777
  chunks into 56,536 records (-76%, 10.6 -> 2.6 MiB), fewer records than
  AE-Min (97k), RAM (77k), or SeqCDC (71k) emit as plain chunks; on run-free
  data record count is unchanged.
- The benchmark corpus (raw Debian image + 3 DEB .ova files + sha256
  manifest) is public:
  https://mincatcdc-bench-corpus.t3.storage.dev/corpus/MANIFEST.sha256
- Tests: a `packed_repeats` soundness test against the real boundary search as
  oracle (break position swept over every byte at every alignment), a
  segment-stream differential against the pre-SIMD caterpillar (adversarial
  corpus + proptest), per-width SIMD agreement tests, and a streaming corpus
  entry with a period break inside the final decision window.

## 0.5.0 and earlier

See git history (`git log --oneline`).
