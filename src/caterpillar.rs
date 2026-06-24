//! Caterpillar coalescing — a metadata-efficiency layer over the chunkers.
//!
//! PROTOTYPE (not upstream). Inspired by the "caterpillar" idea in the Chonkers
//! algorithm (Berger, 2025): periodic / low-entropy regions make any CDC emit a
//! flood of tiny, near-`min_size` chunks. Each chunk is a separate metadata
//! record (e.g. one `extent_chunks` row), so a long zero-fill or repeated-record
//! region costs metadata out of all proportion to its information content.
//!
//! This layer wraps a [`SliceChunker`] and run-length-encodes *maximal runs of
//! byte-identical adjacent chunks* into a single [`Segment::Caterpillar`]. On a
//! zero-fill / constant / exact-repeat region this collapses N records into one
//! `(unit, count)` record. On generic data it is a no-op (adjacent chunks differ,
//! every segment is a [`Segment::Solo`]) and adds ~one slice comparison per chunk.
//!
//! ## What it catches and what it does not
//! mincdc places a boundary at the minimizing window in `[min, max]`. On periodic
//! data of period `P`, consecutive chunks have a *constant phase shift* equal to
//! `chunk_len % P`. When that is `0` (zero-fill `P=1`, constant bytes, or a period
//! that divides the chunk length) the chunks are byte-identical and coalesce
//! perfectly. When the phase rotates (`chunk_len % P != 0`) the chunks are
//! rotations of the same period, *not* byte-equal, so this simple
//! content-equality RLE does **not** coalesce them — a period-detecting variant
//! would. The tests below measure exactly where the line falls.

use crate::{Cdc, Chunk, SliceChunker};

/// One output unit of [`CaterpillarChunker`].
#[derive(Debug)]
pub enum Segment<'a> {
    /// A single chunk whose neighbor differed — emitted as-is.
    Solo(Chunk<'a>),
    /// A maximal run of `count` (>= 2) byte-identical adjacent chunks, stored as
    /// the repeated `unit` plus the repeat `count`.
    Caterpillar {
        /// Start offset of the run within the input.
        offset: usize,
        /// The repeated chunk's bytes (one period of the run).
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
}

/// Wraps a [`SliceChunker`] and run-length-encodes runs of byte-identical
/// adjacent chunks into [`Segment::Caterpillar`]s.
pub struct CaterpillarChunker<'a, C> {
    data: &'a [u8],
    inner: SliceChunker<'a, C>,
    /// A chunk already pulled from `inner` that belongs to the next run.
    carry: Option<Chunk<'a>>,
}

impl<'a, C: Cdc> CaterpillarChunker<'a, C> {
    /// Creates a new caterpillar-coalescing chunker over `bytes`.
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
        // Start the run with a carried chunk, or pull a fresh one.
        let first = self.carry.take().or_else(|| self.inner.next())?;
        let offset = first.offset();
        // Re-slice the backing data so `unit` carries the full 'a lifetime
        // (Chunk's Deref only yields a borrow tied to `first`).
        let unit: &'a [u8] = &self.data[offset..offset + first.len()];

        let mut count = 1usize;
        loop {
            match self.inner.next() {
                Some(c) if &*c == unit => count += 1,
                Some(c) => {
                    self.carry = Some(c);
                    break;
                }
                None => break,
            }
        }

        if count == 1 {
            Some(Segment::Solo(first))
        } else {
            Some(Segment::Caterpillar { offset, unit, count })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MinCdcHash4;

    const MIN: usize = 2 * 1024;
    const MAX: usize = 14 * 1024;

    /// Returns (plain_chunks, caterpillar_segments, expanded_chunks) and asserts
    /// the caterpillar output preserves coverage and reconstructs the input.
    fn measure(label: &str, data: &[u8]) -> (usize, usize, usize) {
        let cdc = MinCdcHash4::new();
        let plain = SliceChunker::new(data, MIN, MAX, cdc).count();

        let mut segs = 0usize;
        let mut expanded = 0usize;
        let mut next_off = 0usize;
        let mut rebuilt: Vec<u8> = Vec::with_capacity(data.len());
        for s in CaterpillarChunker::new(data, MIN, MAX, cdc) {
            assert_eq!(s.offset(), next_off, "{label}: segment offset not contiguous");
            segs += 1;
            expanded += s.chunk_count();
            match &s {
                Segment::Solo(c) => rebuilt.extend_from_slice(c),
                Segment::Caterpillar { unit, count, .. } => {
                    for _ in 0..*count {
                        rebuilt.extend_from_slice(unit);
                    }
                }
            }
            next_off += s.len();
        }
        assert_eq!(rebuilt, data, "{label}: caterpillar must reconstruct input exactly");
        assert_eq!(expanded, plain, "{label}: must represent the same underlying chunks");

        let reduction = if plain > 0 {
            100.0 * (1.0 - segs as f64 / plain as f64)
        } else {
            0.0
        };
        println!("{label:<26} plain={plain:>5}  caterpillar={segs:>5}  records -{reduction:>5.1}%");
        (plain, segs, expanded)
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

    #[test]
    fn caterpillar_wins_on_low_entropy() {
        // Zero-fill (period 1): every chunk is identical -> collapses to ~1 record.
        let (plain, segs, _) = measure("zero-fill 1MiB", &vec![0u8; 1024 * 1024]);
        assert!(segs * 10 < plain, "zero-fill should collapse drastically");

        // Constant non-zero byte: same story.
        let (plain, segs, _) = measure("const 0xAB 1MiB", &vec![0xABu8; 1024 * 1024]);
        assert!(segs * 10 < plain);

        // Exact-repeat of a small unit (period divides nicely for some lengths).
        let unit = xorshift(7, 512);
        let mut repeated = Vec::new();
        for _ in 0..2000 {
            repeated.extend_from_slice(&unit);
        }
        measure("repeat 512B x2000", &repeated);

        // Long zero-run embedded in random data.
        let mut mixed = xorshift(99, 1024 * 1024);
        for b in mixed.iter_mut().take(768 * 1024).skip(256 * 1024) {
            *b = 0;
        }
        measure("random+zero-run", &mixed);
    }

    #[test]
    fn caterpillar_is_noop_on_random() {
        // On incompressible data, adjacent chunks differ: segments ~= plain chunks,
        // so the layer is essentially free and never hurts.
        let data = xorshift(1234, 1024 * 1024);
        let (plain, segs, _) = measure("random 1MiB", &data);
        assert!(
            segs as f64 >= plain as f64 * 0.98,
            "should not coalesce meaningfully on random data"
        );
    }

    #[test]
    fn phase_rotating_period_limitation() {
        // A period that does NOT divide the chunk length: chunks are rotations of
        // the same period, not byte-equal, so content-equality RLE can't coalesce.
        // Documents the limitation honestly (a period-detecting variant would win).
        let period: Vec<u8> = (0..777u32).map(|i| (i % 251) as u8).collect();
        let mut data = Vec::new();
        while data.len() < 1024 * 1024 {
            data.extend_from_slice(&period);
        }
        measure("phase-rotating period", &data);
    }
}
