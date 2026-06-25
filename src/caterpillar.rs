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
//! [`CaterpillarChunker`] works on an in-memory byte slice (it wraps
//! [`SliceChunker`]). For inputs larger than memory, [`CaterpillarReadChunker`]
//! does the same coalescing over a streaming [`ReadChunker`] in bounded memory,
//! yielding borrowed [`Segment`]s (valid until the next call, like
//! [`ReadChunker::next`](crate::ReadChunker)). It is tier 1 only and never
//! copies; a run longer than the internal buffer is emitted as several segments
//! rather than one (still far fewer records than one-per-chunk).
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
//! (A second tier — content-defined *period detection* for phase-rotating runs —
//! was evaluated and removed: mincdc self-aligns to most periods so it rarely
//! helped, and it cost a lot per chunk. See `examples/CATBENCH_RESULTS.md` and
//! the `proto/caterpillar-period` branch for the full implementation + numbers.)

use std::io::{self, Read};

use crate::{Cdc, Chunk, SliceChunker};

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
/// A run longer than the internal buffer is emitted as several segments instead
/// of one (still far fewer records than one-per-chunk, and every segment's
/// [`dedup_key`](Segment::dedup_key) is content-defined, so they dedup to one
/// stored blob). Where those splits land depends on the reader's `read()` sizes,
/// so the *record grouping* is not reproducible across readers — but the stored
/// content is, which is what content-addressed dedup relies on. Tier 1 only
/// (no period detection in the streaming path).
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
        self.ensure()?;
        if self.unread == 0 {
            self.done = true;
            return Ok(None);
        }

        let base = self.buf_offset;
        let base_stream = self.stream_offset;
        // Reuse the boundary the previous call already computed, if any.
        let unit_len = match self.pending_len.take() {
            Some(l) => l,
            None => self
                .chunk_len(base, self.unread)
                .expect("buffer was ensured but the first chunk could not be decided"),
        };

        // Coalesce identical adjacent chunks within the buffered region (no refill
        // mid-run, so the unit borrow stays valid).
        let mut run_len = unit_len;
        let mut count = 1usize;
        loop {
            let cur = base + run_len;
            let avail = self.unread - run_len;
            match self.chunk_len(cur, avail) {
                Some(nl)
                    if nl == unit_len
                        && self.buf[cur..cur + nl] == self.buf[base..base + unit_len] =>
                {
                    count += 1;
                    run_len += nl;
                },
                // Next chunk differs: stash its already-computed length so the
                // next call doesn't recompute the boundary.
                Some(nl) => {
                    self.pending_len = Some(nl);
                    break;
                },
                // Can't decide without a refill: recompute next call.
                None => {
                    self.pending_len = None;
                    break;
                },
            }
        }

        self.buf_offset += run_len;
        self.unread -= run_len;
        self.stream_offset += run_len;

        let seg = if count >= 2 {
            Segment::Caterpillar {
                offset: base_stream,
                unit: &self.buf[base..base + unit_len],
                count,
            }
        } else {
            Segment::Solo(Chunk::new(&self.buf[base..base + unit_len], base_stream))
        };
        Ok(Some(seg))
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
