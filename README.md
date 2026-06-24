# mincatcdc

> A fork of [MinCDC](https://github.com/orlp/mincdc) that adds a *caterpillar*
> layer for metadata-efficient content-defined chunking on redundant data. The
> algorithm described below is MinCDC; see [This fork](#this-fork-mincatcdc) for
> what's added and why.

MinCDC is a very simple yet efficient content-defined chunking algorithm. It
splits your input data into chunks in such a way that the boundaries are defined
by the data itself. This means duplicate regions in large (sets of) files are
likely to have identical boundaries and can thus efficiently be found and
deduplicated.

To start using `mincdc` add the following to your `Cargo.toml`:

    [dependencies]
    mincdc = "0.1"

Please refer to [the documentation](https://docs.rs/mincdc) for more information
on usage. 


## This fork: mincatcdc

This is a fork of [orlp/mincdc](https://github.com/orlp/mincdc) that adds one
thing on top: a **caterpillar** layer.

**The problem it fixes.** Like any content-defined chunker, mincdc turns a long
run of identical bytes — zeros, padding, a repeated block — into a flood of tiny
chunks. Each chunk is a metadata record (a fingerprint, an index entry). The
data dedupes to almost nothing, but you still pay to track all those records. A
mostly-empty 200 MiB disk image, for example, becomes ~182,000 records.

**What the caterpillar does.** It is a small, lossless pass over the chunk stream
that collapses any run of byte-identical adjacent chunks into a single record
with a repeat count. On that disk image: **182,701 → 7,798 records (−96%)**, with
no change to what is stored and no preprocessing required. On normal data it does
nothing and costs nothing (it is a no-op when there are no runs), so it keeps
mincdc's speed and deduplication everywhere else.

**Lineage.** The idea comes from the [Chonkers
algorithm](https://arxiv.org/abs/2509.11121) (Berger, 2025), which calls a
periodic run a *caterpillar*. We borrowed just that one practical idea and put it
on top of mincdc rather than adopting the whole (heavier) Chonkers machinery.

**Why it's nice.** It's a drop-in wrapper that gives you metadata-efficient CDC
on redundant data without writing any domain-specific preprocessing (zero
detection, sparse reads, etc.). See `examples/CATBENCH_RESULTS.md` and
`examples/REALBENCH_RESULTS.md` for measurements on real corpora.

**How the disk-image number was measured.** Two 200 MiB APFS images were created
with `hdiutil` (`hdiutil create -size 200m -fs APFS ...`), each holding a real
source tree (the second also holds an extra version), so each image is ~92%
written-zero free space — not sparse holes, so `SEEK_HOLE` would not skip them.
Both images were chunked with `cargo run --release --example catbench` at
`min=2048, max=14336`. Plain mincdc produced 182,701 records; the caterpillar
produced 7,798, with identical deduplicated content. Full method and the other
corpora (Linux kernels, containers, SQLite, source trees) are in
`examples/REALBENCH_RESULTS.md`.

This fork also fixes a soundness bug in the upstream SIMD prefetch and adds test
coverage (cross-SIMD-width determinism, an invariant/oracle harness).


## Algorithm

The basic idea of MinCDC is to choose chunk boundaries based on the minimum
value of a sliding window over the input data. That is, if the desired chunk
size is between `min_size` and `max_size`, we find some `min_size <= i <=
max_size` such that `evaluate(bytes[i - w..i])` is minimized, where `w` is the
window size, breaking ties by choosing the earliest such `i`. Then we return
chunk `bytes[..i]` and repeat the process on the remainder `bytes[i..]`.

This library provides two SIMD-accelerated implementations of MinCDC, both with
a window size of 4:

 - `MinCDC4`, where the evaluation function is
   `u32::from_le_bytes(bytes[i - 4..i])`, i.e. a window size of 4 bytes
   interpreting the bytes as a little-endian `u32`, and
 - `MinCDCHash4`, where the evaluation function is
   `hash(u32::from_le_bytes(bytes[i - 4..i]))`. The hash function used is
   the very simple `hash(x) = x.wrapping_mul(a).wrapping_add(b)`, for
   some constants `a` and `b`.

**`MinCDCHash4` can be slightly (~10%) slower but is far more robust and
predictable, it is the recommended default**.

   
## Performance

MinCDC is several times faster than the commonly used
[FastCDC](https://crates.io/crates/fastcdc) while providing a similar amount of
deduplicating power. To benchmark this I downloaded all available Linux kernel
6.x.tar archives (`tools/download-linux.sh`) and ran the below algorithms on
them, all of them configured to target an expected chunk size of 8 KiB.

To determine the chunking speed I only chunked `linux-6.0.tar` while the file
was loaded into memory to avoid disk overhead. The dedup% is one minus the total
size of unique chunks divided by the total size of all input files (thus higher
is better). The normalized dedup% is the same percentage acquired from repeating
the experiment with different window sizes until the mean chunk size matched 8
KiB (+/- 1%). **This is important when comparing deduplication power since
smaller chunks typically means better deduplication.**

| Algorithm     |   AMD 9950X | Apple M2 Pro | Dedup% | Mean Chunk Size | Dedup% (normalized) |
| --------------|-------------|--------------|--------|-----------------|---------------------|
| MinCDCHash4-s | 41.3 GB / s |  23.8 GB / s | 61.08% |            8015 |              60.92% |
| MinCDCHash4-l | 44.5 GB / s |  15.7 GB / s | 61.57% |            8221 |              61.57% |
| MinCDC4-s     | 41.7 GB / s |  26.1 GB / s | 62.11% |            7383 |              60.52% |
| MinCDC4-l     | 42.0 GB / s |  16.9 GB / s | 64.51% |            6436 |              60.69% |
| FastCDC-s     |  6.6 GB / s |   4.1 GB / s | 54.38% |           12866 |              61.81% |
| FastCDC-l     |  5.2 GB / s |   3.2 GB / s | 54.87% |           12764 |              ~ 62%* |

Here the "-s" variants use a small window size of 8 KiB +/- 25%
(min=6144, max=10240), and the "-l" variants use a larger window size of
8 KiB +/- 50% (min=4096, max=12288). The maximum chunk size was increased
further for FastCDC as it inherently has a long tail of chunk sizes (see below),
this did not impact chunking speed much.

The normalized dedup% for FastCDC-l is marked with an asterisk because I was
unable to get the mean chunk size within 1% of 8KiB. The mean size would
suddenly jump from 7741 to 10846 just by making a tiny adjustment in window
size. For comparison, MinCDCHash4-l with a mean size of 7741 has a dedup% of
62.49% versus FastCDC-l's 62.75%.

## Chunk Distribution

Unlike most other content-defined chunking algorithms, the distribution of chunk
sizes generated by MinCDCHash4 is almost entirely uniform in the range
`min_size`, `max_size`. This makes it very predictable and well-behaved; the
expected chunk size is also very close to the mean chunk size. Compare that with
FastCDC's distribution for an expected chunk size of 8 KiB:

| MinCDCHash4 | FastCDC |
|-------------|---------|
| <img src="assets/mincdc-chunk-size-distr.png" width=300> | <img src="assets/fastcdc-chunk-size-distr.png" width=300>

While the peak is at 8 KiB as expected for FastCDC, there is a long and heavy
tail, increasing the mean chunk size by a lot. MinCDCHash4 never creates a
chunk outside of the specified range, except for the last chunk which may be
smaller.

There is still a bias towards smaller chunks as MinCDC breaks ties in the
minimum value towards the earlier breakpoint, but this bias is relatively small.

