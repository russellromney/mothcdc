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
//! Inside a run the caterpillar does not pay for chunking at all: a
//! VectorCDC-style *packed scan* (whole-vector SIMD equality — broadcast
//! compare for constant bytes, self-shifted compare for longer periods) proves
//! the region periodic once, and every chunk whose boundary decision provably
//! repeats is emitted without re-running the boundary search. Redundant
//! regions therefore chunk *faster* than plain mincdc instead of slightly
//! slower (see `packed_repeats` for the proof and `benches/throughput.rs` for
//! numbers).
//!
//! [`CaterpillarChunker`] works on an in-memory byte slice (it wraps
//! [`SliceChunker`]). For inputs larger than memory, [`CaterpillarReadChunker`]
//! does the same coalescing over a streaming [`ReadChunker`] in bounded memory,
//! yielding borrowed [`Segment`]s (valid until the next call, like
//! [`ReadChunker::next`](crate::ReadChunker)). Runs coalesce across buffer
//! refills (a run's unit is copied once — at most `max_size` bytes — when it
//! crosses a refill), so even a run far longer than the buffer is one record;
//! everything else is zero-copy.
//!
//! # Example
//! ```
//! use mothcdc::{MinCdcHash4, caterpillar::{CaterpillarChunker, Segment}};
//!
//! let data = vec![0u8; 64 * 4096]; // a long zero run
//! let segs: Vec<_> = CaterpillarChunker::new(&data, 4096, 12288, MinCdcHash4::new()).collect();
//!
//! // The whole zero run collapses into one record instead of ~64 chunks.
//! assert_eq!(segs.len(), 1);
//! assert!(matches!(segs[0], Segment::Caterpillar { .. }));
//! // `dedup_key()` gives the unique bytes to fingerprint/store, regardless of variant.
//! assert!(segs[0].dedup_key().iter().all(|&b| b == 0));
//! ```
//!
//! (A second tier — content-defined *period detection* for phase-rotating runs —
//! was evaluated and removed: mincdc self-aligns to most periods so it rarely
//! helped, and it cost a lot per chunk. See `examples/CATBENCH_RESULTS.md` and
//! the `proto/caterpillar-period` branch for the full implementation + numbers.)

use std::io::{self, Read};

use crate::{Cdc, Chunk, SliceChunker, simd};

/// VectorCDC-style packed scanning: the fast-forward that lets the caterpillar
/// skip the boundary search inside repetitive runs.
///
/// Given that a chunk of length `u` starts at the beginning of `tail`
/// (`tail[..u]` is that chunk, and `tail` extends to the end of decidable
/// data), returns how many *additional* chunks — each exactly `u` bytes and
/// byte-identical to the first — are guaranteed to follow, without running the
/// argmin boundary search (Phase 1) for any of them.
///
/// Why this is sound: `next_chunk_len` for a chunk starting at `p` is a pure
/// function of the decision window `tail[p..p + max_size]`, provided at least
/// `max_size + 1` bytes remain (which rules out its truncated / short-final
/// branches). One packed equality scan computes `e`, the largest extent such
/// that `tail[i] == tail[i - u]` for all `u <= i < e` — i.e. the data is
/// periodic with period `u` over `tail[..e]`. For any chunk start `p = k * u`
/// with `p + max_size <= e`, every byte of its decision window equals the byte
/// one period earlier, so by induction the window is byte-identical to the
/// first chunk's window and the boundary search *must* return `u` again. Those
/// chunks are emitted at packed-scan speed; the first chunk whose window
/// leaves the periodic region (or the decidable region: the `tail.len() - 1`
/// bound) falls back to the normal argmin + compare path.
///
/// The handoff between the two phases is therefore zero-state: this function
/// only ever *pre-pays* boundary decisions that Phase 1 would provably make,
/// so the caller resumes the ordinary per-chunk loop at `(1 + repeats) * u` as
/// if those chunks had been computed one by one.
pub(crate) fn packed_repeats(tail: &[u8], u: usize, max_size: usize) -> usize {
    let n = tail.len();
    if u == 0 || u >= n {
        return 0;
    }
    let unit = &tail[..u];
    // Extent of the periodic region. A constant-byte unit (zero-fill, padding)
    // uses the broadcast form: one splat + one load + one packed compare per
    // vector; its scan cost is bounded by the actual run of that byte. The
    // `unit[u - 1] == unit[0]` pre-check makes the constant-unit detection
    // O(1) on data without runs.
    let e = if unit[u - 1] == unit[0] && simd::byte_run_len(unit, unit[0]) == u {
        u + simd::byte_run_len(&tail[u..], unit[0])
    } else {
        // Staged scan. Data like tar archives of similar trees is full of
        // medium pseudo-periodic stretches (similar adjacent files, sparse
        // padding) where an unbounded extent scan runs long and then yields
        // zero usable repeats — measured as a 31% throughput loss on Linux
        // kernel source tars. So first probe exactly one unit: if the next
        // chunk is not byte-identical to this one, no coalescing is possible
        // (K = 0 either way) and the probe cost is bounded by the mismatch
        // position, exactly like the pre-packed-scan memcmp. Only a proven
        // identical neighbor unlocks the full extent scan.
        if 2 * u > n {
            return 0;
        }
        let probe = simd::common_prefix_len(&tail[u..2 * u], unit);
        if probe < u {
            return 0;
        }
        2 * u + simd::common_prefix_len(&tail[2 * u..], &tail[u..n - u])
    };
    // A chunk needing bytes past `n - 1` could hit `next_chunk_len`'s truncated
    // (end-of-data) branches, whose result depends on more than the window
    // bytes; leave it to the slow path.
    let lim = e.min(n - 1);
    lim.saturating_sub(max_size) / u
}

/// One output unit of [`CaterpillarChunker`] or [`CaterpillarReadChunker`].
///
/// The borrow source differs by producer: from [`CaterpillarChunker`] a segment
/// borrows the input slice (valid for the whole iteration); from
/// [`CaterpillarReadChunker`] it borrows the reader's reused buffer and is valid
/// **only until the next call**. Process it (or copy [`dedup_key`](Self::dedup_key))
/// before advancing the streaming chunker.
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
}

impl<'a> Segment<'a> {
    /// Start offset of this segment within the input.
    pub fn offset(&self) -> usize {
        match self {
            Segment::Solo(c) => c.offset(),
            Segment::Caterpillar { offset, .. } => *offset,
        }
    }

    /// Number of underlying chunks represented (1 for [`Segment::Solo`]).
    pub fn chunk_count(&self) -> usize {
        match self {
            Segment::Solo(_) => 1,
            Segment::Caterpillar { count, .. } => *count,
        }
    }

    /// Total number of bytes covered by this segment.
    pub fn len(&self) -> usize {
        match self {
            Segment::Solo(c) => c.len(),
            Segment::Caterpillar { unit, count, .. } => unit.len() * count,
        }
    }

    /// Whether this segment covers zero bytes (never true in practice).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The unique content to fingerprint and store for content addressing: the
    /// chunk bytes ([`Segment::Solo`]) or the repeated unit
    /// ([`Segment::Caterpillar`]).
    ///
    /// This is also exactly the bytes to tile back to [`len`](Self::len) to
    /// reconstruct the segment, so a store-then-restore round-trip is lossless:
    /// hash and store `dedup_key()`, record [`offset`](Self::offset) and
    /// [`len`](Self::len), and on restore tile the stored bytes to `len`. See
    /// [`reconstruct_into`](Self::reconstruct_into).
    pub fn dedup_key(&self) -> &[u8] {
        match self {
            Segment::Solo(c) => c,
            Segment::Caterpillar { unit, .. } => unit,
        }
    }

    /// Appends this segment's original bytes to `out` (the inverse of chunking):
    /// the chunk for [`Segment::Solo`] or the unit repeated `count` times for
    /// [`Segment::Caterpillar`]. Equivalent to tiling [`dedup_key`](Self::dedup_key)
    /// to [`len`](Self::len).
    pub fn reconstruct_into(&self, out: &mut Vec<u8>) {
        let key = self.dedup_key();
        let total = self.len();
        let mut written = 0;
        while written < total {
            let take = key.len().min(total - written);
            out.extend_from_slice(&key[..take]);
            written += take;
        }
    }
}

/// Wraps a [`SliceChunker`] and coalesces periodic / repeated regions.
pub struct CaterpillarChunker<'a, C> {
    data: &'a [u8],
    inner: SliceChunker<'a, C>,
    carry: Option<Chunk<'a>>,
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
        }
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

        // Packed-scanning fast path: prove the region periodic once and emit
        // every chunk whose decision window stays inside it, skipping the
        // argmin boundary search entirely (see `packed_repeats`). The skipped
        // chunks were never pulled from `inner`, so jump it forward; the loop
        // below then continues the run (or ends it) through the normal path.
        let repeats = packed_repeats(&self.data[start..], first_len, self.inner.max_size);
        if repeats > 0 {
            count += repeats;
            self.inner.offset = start + count * first_len;
        }

        let pending: Option<Chunk<'a>> = loop {
            match self.inner.next() {
                Some(c) if &*c == unit => count += 1,
                other => break other,
            }
        };
        self.carry = pending;
        if count >= 2 {
            Some(Segment::Caterpillar {
                offset: start,
                unit,
                count,
            })
        } else {
            Some(Segment::Solo(first))
        }
    }
}

/// Streaming caterpillar: [`CaterpillarChunker`] for a [`Read`] source.
///
/// It coalesces byte-identical adjacent chunks *inside* the reader's buffer and
/// yields **borrowed** [`Segment`]s valid only until the next call (the same
/// contract as [`ReadChunker::next`](crate::ReadChunker)) — so it never copies
/// and runs in bounded memory, working on inputs larger than RAM.
///
/// A run is coalesced even when it is longer than the internal buffer: when a
/// run reaches the end of the buffered data, its unit is copied once (at most
/// `max_size` bytes per run) and counting continues across refills, so a
/// multi-gigabyte zero region is still a single record. Segments therefore
/// group the same way regardless of the reader's `read()` sizes. Everything
/// else stays zero-copy: solos and runs contained in one buffer borrow the
/// internal buffer directly. Tier 1 only (no period detection in the
/// streaming path).
pub struct CaterpillarReadChunker<R, C> {
    min_size: usize,
    max_size: usize,
    cdc: C,
    reader: R,
    buf: Vec<u8>,
    buf_offset: usize,
    unread: usize,
    stream_offset: usize,
    eof: bool,
    done: bool,
    /// Length of the chunk at `buf_offset` if already computed by a previous
    /// call's lookahead — avoids recomputing the boundary (a SIMD scan) twice.
    pending_len: Option<usize>,
    /// A run continued across buffer refills: (owned unit bytes, stream offset
    /// of the run start, chunks counted so far — always >= 2).
    carry_run: Option<(Vec<u8>, usize, usize)>,
    /// Owns the unit of a carried run while its borrowed [`Segment`] is live.
    emit_unit: Vec<u8>,
}

impl<R, C: Cdc> CaterpillarReadChunker<R, C> {
    /// Creates a zero-copy streaming caterpillar chunker.
    pub fn new(reader: R, min_size: usize, max_size: usize, cdc: C) -> Self {
        assert!(min_size <= max_size && max_size > 0);
        let buf_size = crate::MIN_BUFFER_SIZE + (max_size + 1) + min_size * 4;
        Self {
            min_size,
            max_size,
            cdc,
            reader,
            buf: vec![0; buf_size],
            buf_offset: 0,
            unread: 0,
            stream_offset: 0,
            eof: false,
            done: false,
            pending_len: None,
            carry_run: None,
            emit_unit: Vec::new(),
        }
    }

    /// Length of the chunk starting at buffer position `p` given `avail` buffered
    /// bytes from there, or `None` if it can't be decided without more data.
    /// Delegates to the shared [`crate::next_chunk_len`] so it can't diverge from
    /// `SliceChunker` / `ReadChunker`.
    fn chunk_len(&self, p: usize, avail: usize) -> Option<usize> {
        crate::next_chunk_len(
            &self.buf[p..p + avail],
            self.min_size,
            self.max_size,
            self.eof,
            &self.cdc,
        )
    }

    /// Coalesce byte-identical chunks starting at buffer position `base`, whose
    /// first (already decided) chunk is `buf[base..base + u]`. Returns
    /// `(bytes covered, chunks counted, boundary of the next chunk)`, where the
    /// boundary is `Some(len)` if a differing chunk was decided and `None` if
    /// it is undecidable without more data.
    fn coalesce_from(&self, base: usize, u: usize) -> (usize, usize, Option<usize>) {
        // Packed-scanning fast path: guaranteed repeats within the buffered
        // region cost one equality scan instead of one boundary search each.
        let repeats = packed_repeats(&self.buf[base..base + self.unread], u, self.max_size);
        let mut run_len = u + repeats * u;
        let mut count = 1 + repeats;
        loop {
            let cur = base + run_len;
            let avail = self.unread - run_len;
            match self.chunk_len(cur, avail) {
                Some(nl) if nl == u && self.buf[cur..cur + nl] == self.buf[base..base + u] => {
                    count += 1;
                    run_len += nl;
                },
                other => return (run_len, count, other),
            }
        }
    }
}

impl<R: Read, C: Cdc> CaterpillarReadChunker<R, C> {
    /// Reads/shifts until at least `max_size + 1` bytes are buffered (or EOF).
    /// Only ever called at a run boundary, when no segment borrow is live.
    fn ensure(&mut self) -> io::Result<()> {
        let need = self.max_size + 1;
        while !self.eof && self.unread < need {
            if self.buf.len() - self.buf_offset < need {
                self.buf
                    .copy_within(self.buf_offset..self.buf_offset + self.unread, 0);
                self.buf_offset = 0;
            }
            let n = self
                .reader
                .read(&mut self.buf[self.buf_offset + self.unread..])?;
            if n == 0 {
                self.eof = true;
                break;
            }
            self.unread += n;
        }
        Ok(())
    }

    /// Gets the next [`Segment`] (borrowed, valid until the next call), or `None`.
    ///
    /// Like [`ReadChunker`](crate::ReadChunker), a `read` returning `Ok(0)` is
    /// treated as end of input.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> io::Result<Option<Segment<'_>>> {
        if self.done {
            return Ok(None);
        }
        loop {
            self.ensure()?;
            let base = self.buf_offset;

            if self.unread == 0 {
                self.done = true;
                // The stream ended exactly at a carried run's boundary.
                return Ok(self.carry_run.take().map(|(unit, off, count)| {
                    self.emit_unit = unit;
                    Segment::Caterpillar {
                        offset: off,
                        unit: &self.emit_unit,
                        count,
                    }
                }));
            }

            // A run carried across a refill: does it continue at the front of
            // the freshly filled buffer?
            if let Some((unit, run_off, carried)) = self.carry_run.take() {
                let u = unit.len();
                match self.chunk_len(base, self.unread) {
                    Some(l) if l == u && self.buf[base..base + l] == unit[..] => {
                        // It continues; coalesce through this buffer too.
                        let (run_len, more, pend) = self.coalesce_from(base, u);
                        self.buf_offset += run_len;
                        self.unread -= run_len;
                        self.stream_offset += run_len;
                        let count = carried + more;
                        if pend.is_none() && !self.eof {
                            // Ran out of buffer again mid-run: keep carrying.
                            self.carry_run = Some((unit, run_off, count));
                            continue;
                        }
                        self.pending_len = pend;
                        self.emit_unit = unit;
                        return Ok(Some(Segment::Caterpillar {
                            offset: run_off,
                            unit: &self.emit_unit,
                            count,
                        }));
                    },
                    // The run ended at the refill boundary: emit it and stash
                    // the just-computed boundary (if any) for the next call.
                    boundary => {
                        self.pending_len = boundary;
                        self.emit_unit = unit;
                        return Ok(Some(Segment::Caterpillar {
                            offset: run_off,
                            unit: &self.emit_unit,
                            count: carried,
                        }));
                    },
                }
            }

            // Fresh segment. Reuse the boundary a previous call computed, if any.
            let base_stream = self.stream_offset;
            let unit_len = match self.pending_len.take() {
                Some(l) => l,
                None => self
                    .chunk_len(base, self.unread)
                    .expect("buffer was ensured but the first chunk could not be decided"),
            };

            let (run_len, count, pend) = self.coalesce_from(base, unit_len);
            self.buf_offset += run_len;
            self.unread -= run_len;
            self.stream_offset += run_len;

            if pend.is_none() && !self.eof && count >= 2 {
                // The run reached the end of the buffered data and may continue
                // after a refill: copy the unit (once per crossing run) and
                // keep counting instead of splitting the record.
                self.carry_run =
                    Some((self.buf[base..base + unit_len].to_vec(), base_stream, count));
                continue;
            }
            self.pending_len = pend;

            let seg = if count >= 2 {
                Segment::Caterpillar {
                    offset: base_stream,
                    unit: &self.buf[base..base + unit_len],
                    count,
                }
            } else {
                Segment::Solo(Chunk::new(&self.buf[base..base + unit_len], base_stream))
            };
            return Ok(Some(seg));
        }
    }
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

    /// Drives the caterpillar, asserts exact reconstruction + same underlying
    /// chunk count as plain mincdc, and returns (plain_chunks, records).
    fn measure(label: &str, data: &[u8], min: usize, max: usize) -> (usize, usize) {
        let cdc = MinCdcHash4::new();
        let plain = SliceChunker::new(data, min, max, cdc).count();

        let mut records = 0usize;
        let mut expanded = 0usize;
        let mut next_off = 0usize;
        let mut rebuilt: Vec<u8> = Vec::with_capacity(data.len());
        for s in CaterpillarChunker::new(data, min, max, cdc) {
            assert_eq!(s.offset(), next_off, "{label}: offset not contiguous");
            records += 1;
            expanded += s.chunk_count();
            s.reconstruct_into(&mut rebuilt);
            next_off += s.len();
        }
        assert_eq!(rebuilt, data, "{label}: must reconstruct input exactly");
        assert_eq!(expanded, plain, "{label}: must represent same chunk count");
        (plain, records)
    }

    #[test]
    fn collapses_low_entropy_and_is_noop_on_random() {
        // Zero-fill collapses to ~one record.
        let (plain, records) = measure("zero-fill 1MiB", &vec![0u8; 1024 * 1024], MIN, MAX);
        assert!(records * 10 < plain, "zero-fill should collapse");

        // Random has no adjacent-identical runs: a lossless no-op.
        let data = xorshift(1234, 1024 * 1024);
        let (plain, records) = measure("random 1MiB", &data, MIN, MAX);
        assert!(
            records as f64 >= plain as f64 * 0.98,
            "must not coalesce random"
        );
    }

    /// The pre-SIMD caterpillar, kept as a differential oracle: pull every
    /// chunk from the plain chunker and RLE-coalesce byte-identical neighbors.
    /// The packed-scanning fast path must produce exactly this segment stream —
    /// same grouping, same offsets, same unit bytes.
    fn reference_segments(data: &[u8], min: usize, max: usize) -> Vec<(usize, Vec<u8>, usize)> {
        let mut out: Vec<(usize, Vec<u8>, usize)> = Vec::new();
        for c in SliceChunker::new(data, min, max, MinCdcHash4::new()) {
            match out.last_mut() {
                Some((_, unit, count)) if unit[..] == c[..] => *count += 1,
                _ => out.push((c.offset(), c.to_vec(), 1)),
            }
        }
        out
    }

    fn assert_matches_reference(label: &str, data: &[u8], min: usize, max: usize) {
        let got: Vec<(usize, Vec<u8>, usize)> =
            CaterpillarChunker::new(data, min, max, MinCdcHash4::new())
                .map(|s| match s {
                    Segment::Solo(c) => (c.offset(), c.to_vec(), 1),
                    Segment::Caterpillar {
                        offset,
                        unit,
                        count,
                    } => (offset, unit.to_vec(), count),
                })
                .collect();
        let want = reference_segments(data, min, max);
        assert_eq!(got, want, "{label} (min={min} max={max})");
    }

    /// Soundness of `packed_repeats` against its spec, with the real boundary
    /// search as the oracle: every claimed repeat must be exactly the chunk
    /// the slow path would produce — same length, same bytes, under both the
    /// slice (eof) and streaming (non-eof) decision branches.
    ///
    /// The break position sweeps every byte of small periodic buffers, so the
    /// periodic-extent edge and the repeat-count cutoff are exercised at every
    /// alignment. Small windows make "the byte just past the periodic region
    /// wins the argmin" a frequent event rather than a rare one — that is the
    /// case a random corpus almost never samples, and exactly where an
    /// extent/count off-by-one becomes a real boundary divergence.
    #[test]
    fn packed_repeats_claims_only_provable_chunks() {
        let cdc = MinCdcHash4::new();
        for period in [1usize, 2, 3, 5, 8, 13] {
            let unit = xorshift(period as u64 + 40, period);
            let mut base = Vec::new();
            while base.len() < 2000 {
                base.extend_from_slice(&unit);
            }
            base.truncate(2000);

            // break_pos == base.len() means "no break" (pure periodic).
            for break_pos in 0..=base.len() {
                let mut data = base.clone();
                if break_pos < data.len() {
                    data[break_pos] ^= 0xA5;
                }
                for (min, max) in [(16usize, 40usize), (16, 16), (20, 24)] {
                    let u0 = crate::next_chunk_len(&data, min, max, true, &cdc)
                        .expect("non-empty input at eof always chunks");
                    let claimed = packed_repeats(&data, u0, max);
                    for k in 1..=claimed {
                        let tail = &data[k * u0..];
                        let ctx = format!(
                            "period={period} break={break_pos} min={min} max={max} k={k}/{claimed} u0={u0}"
                        );
                        assert_eq!(
                            crate::next_chunk_len(tail, min, max, true, &cdc),
                            Some(u0),
                            "claimed chunk diverges from slow path (eof): {ctx}"
                        );
                        assert_eq!(
                            crate::next_chunk_len(tail, min, max, false, &cdc),
                            Some(u0),
                            "claimed chunk diverges from slow path (streaming): {ctx}"
                        );
                        assert_eq!(
                            &tail[..u0],
                            &data[..u0],
                            "claimed chunk bytes differ from unit: {ctx}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn packed_fast_path_matches_reference() {
        const N: usize = 256 * 1024;
        let mut corpora: Vec<(String, Vec<u8>)> = vec![
            ("zeros".into(), vec![0u8; N]),
            ("const-0xAB".into(), vec![0xABu8; N]),
            ("random".into(), xorshift(7, N)),
            ("tiny-zeros".into(), vec![0u8; 100]),
            // Barely more than one decision window: exercises the fast path's
            // `tail.len() - 1` cutoff where almost every chunk is end-affected.
            ("small-zeros".into(), vec![0u8; 20_000]),
        ];
        for period in [1usize, 3, 4, 5, 100, 777, 2048, 2500, 5000] {
            let unit = xorshift(period as u64, period);
            let mut data = Vec::with_capacity(N + period);
            while data.len() < N {
                data.extend_from_slice(&unit);
            }
            data.truncate(N);
            corpora.push((format!("periodic-{period}"), data));
        }
        // Break the period at adversarial positions: inside the final decision
        // window (straddling the fast path's cutoff), right at it, and mid-run.
        {
            let unit = xorshift(99, 777);
            let mut data = Vec::new();
            while data.len() < N {
                data.extend_from_slice(&unit);
            }
            data.truncate(N);
            for pos in [
                N - 1,
                N - 100,
                N - MAX,
                N - MAX - 1,
                N - MAX + 1,
                N / 2,
                MAX,
                MAX + 1,
                MIN,
            ] {
                let mut d = data.clone();
                d[pos] ^= 0xFF;
                corpora.push((format!("periodic-777-break@{pos}"), d));
            }
        }
        // A run embedded in random data: the fast path starts and ends
        // mid-stream, handing off to the argmin path on both sides.
        {
            let mut d = xorshift(3, N);
            for b in &mut d[64 * 1024..192 * 1024] {
                *b = 0;
            }
            corpora.push(("random+zero-hole".into(), d));
        }

        for (label, data) in &corpora {
            for (min, max) in [(MIN, MAX), (2048, 2200), (64, 256), (16, 16), (4, 20)] {
                assert_matches_reference(label, data, min, max);
            }
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 256, ..proptest::prelude::ProptestConfig::default()
        })]

        /// Random periodic data with random prefixes and breaks: the packed
        /// fast path must produce exactly the reference segment stream.
        #[test]
        fn prop_fast_path_matches_reference(
            period in 1usize..600,
            reps in 2usize..64,
            seed in proptest::prelude::any::<u64>(),
            min in 4usize..300,
            extra in 0usize..300,
            prefix in 0usize..500,
            break_at in proptest::option::of(0.0f64..1.0),
        ) {
            let unit = xorshift(seed | 1, period);
            let mut data = xorshift(seed ^ 0xDEAD, prefix);
            for _ in 0..reps {
                data.extend_from_slice(&unit);
            }
            if let Some(f) = break_at {
                let pos = ((data.len() as f64 - 1.0) * f) as usize;
                data[pos] ^= 0xFF;
            }
            assert_matches_reference("prop", &data, min, min + extra);
        }
    }

    /// A run far longer than the internal buffer (4 MiB) must still be a
    /// single record: the streaming caterpillar carries runs across refills.
    #[test]
    fn streaming_run_crossing_refills_is_one_record() {
        use std::io::Cursor;

        // 24 MiB of zeros -> crosses the 4 MiB buffer several times.
        let n = 24 * 1024 * 1024;
        let data = vec![0u8; n];
        let mut rc = CaterpillarReadChunker::new(Cursor::new(&data), MIN, MAX, MinCdcHash4::new());
        let mut records = 0usize;
        let mut chunks = 0usize;
        let mut covered = 0usize;
        while let Some(s) = rc.next().unwrap() {
            records += 1;
            chunks += s.chunk_count();
            covered += s.len();
            assert!(s.dedup_key().iter().all(|&b| b == 0));
        }
        assert_eq!(covered, n);
        let plain = SliceChunker::new(&data, MIN, MAX, MinCdcHash4::new()).count();
        assert_eq!(chunks, plain, "must represent the same underlying chunks");
        assert!(
            records <= 2,
            "a single giant run must not split at buffer refills (got {records} records)"
        );

        // And a run bracketed by random data still carries across refills.
        let mut data = xorshift(11, 64 * 1024);
        data.extend_from_slice(&vec![0u8; 12 * 1024 * 1024]);
        data.extend_from_slice(&xorshift(12, 64 * 1024));
        let mut rc = CaterpillarReadChunker::new(Cursor::new(&data), MIN, MAX, MinCdcHash4::new());
        let (mut records, mut chunks, mut covered) = (0usize, 0usize, 0usize);
        let mut rebuilt = Vec::with_capacity(data.len());
        while let Some(s) = rc.next().unwrap() {
            records += 1;
            chunks += s.chunk_count();
            covered += s.len();
            s.reconstruct_into(&mut rebuilt);
        }
        assert_eq!(covered, data.len());
        assert_eq!(rebuilt, data, "must reconstruct exactly");
        let plain = SliceChunker::new(&data, MIN, MAX, MinCdcHash4::new()).count();
        assert_eq!(chunks, plain);
        // ~64 KiB of random on each side is ~40-70 records; the 12 MiB zero
        // run must contribute ~1, not ~3 (one per 4 MiB refill).
        assert!(
            records < 120,
            "zero run appears to split at refills (got {records} records)"
        );
    }

    #[test]
    fn self_aligns_and_collapses_periodic_data() {
        // mincdc self-aligns boundaries to periods when a period multiple fits in
        // [min, max], so consecutive chunks are byte-identical and tier-1 collapses
        // the region (this is why the removed period-detection tier rarely helped).
        let period = xorshift(42, 777);
        let mut data = Vec::new();
        while data.len() < 4 * 1024 * 1024 {
            data.extend_from_slice(&period);
        }
        let (plain, records) = measure("period777 wide", &data, MIN, MAX);
        assert!(
            records * 20 < plain,
            "tier-1 collapses self-aligned periodic data"
        );
    }
}
