//! MinCDC
//! ------
//! MinCDC is a very simple yet efficient content-defined chunking algorithm.
//! This library contains a SIMD-accelerated implementation of it.
//!
//! The basic idea of MinCDC is to choose chunk boundaries based on the minimum
//! value of a sliding window over the input data. That is, if the desired
//! chunk size is between `min_size` and `max_size`, we find some
//! `min_size <= i <= max_size` such that `evaluate(bytes[i - w..i])` is
//! minimized, where `w` is the window size, breaking ties by choosing the
//! earliest such `i`. Then we return chunk `bytes[..i]` and repeat the process
//! on the remainder `bytes[i..]`.
//!
//! This crate provides two implementations of MinCDC, both with a window size
//! of 4:
//!  - [`MinCdc4`], where the evaluation function is
//!    `u32::from_le_bytes(bytes[i - 4..i])`, i.e. a window size of 4 bytes
//!    interpreting the bytes as a little-endian `u32`, and
//!  - [`MinCdcHash4`], where the evaluation function is
//!    `hash(u32::from_le_bytes(bytes[i - 4..i]))`. The hash function used is
//!    the very simple `hash(x) = x.wrapping_mul(a).wrapping_add(b)`, for
//!    some constants `a` and `b`.
//!
//! **[`MinCdcHash4`] can be slightly (~10%) slower but is far more robust and
//! predictable, it is the recommended default**.
//!
//! # Usage
//!
//! This library provides two chunkers:
//!  - [`SliceChunker`] for chunking a byte slice, and
//!  - [`ReadChunker`] for chunking a reader implementing [`Read`].
//!
//! Both chunkers take a desired minimum and maximum chunk size as well as a
//! [`Cdc`] instance (either [`MinCdc4`] or
//! [`MinCdcHash4`]). Then by iterating over the chunker
//! (or calling [`next()`](ReadChunker::next) in the case of [`ReadChunker`]) you get chunks of type
//! [`Chunk`], which derefs to a byte slice, but also contains the offset of
//! that chunk in the input stream.
//!
//! # Examples
//!
//! ```rust
//! # use mincatcdc::{MinCdcHash4, SliceChunker};
//! let data = b"Hello, world! This is an example of MinCDC chunking.";
//!
//! // Chunks between 8 and 16 bytes, using MinCdcHash4.
//! let mut chunker = SliceChunker::new(data, 8, 16, MinCdcHash4::new());
//! assert_eq!(chunker.next().as_deref(), Some(&b"Hello, world"[..]));
//! assert_eq!(chunker.next().as_deref(), Some(&b"! This is "[..]));
//! assert_eq!(chunker.next().as_deref(), Some(&b"an example "[..]));
//! assert_eq!(chunker.next().as_deref(), Some(&b"of MinCDC chu"[..]));
//! // Last chunk may be smaller than min_size.
//! assert_eq!(chunker.next().as_deref(), Some(&b"nking."[..]));
//! assert!(chunker.next().is_none());
//! ```

#![warn(missing_docs)]

use std::io::{self, Read};
use std::iter::FusedIterator;
use std::ops::Deref;

const DEFAULT_MULTIPLIER: u32 = 0x915f77f5;
const DEFAULT_ADDEND: u32 = 0x34636463;
const MIN_BUFFER_SIZE: usize = 1024 * 1024 * 4;

/// Caterpillar coalescing layer: metadata efficiency on redundant data.
pub mod caterpillar;
pub use caterpillar::{CaterpillarChunker, CaterpillarReadChunker, Segment};

/// C API for embedding in external benchmark harnesses (feature `capi`).
#[cfg(feature = "capi")]
pub mod capi;

pub(crate) mod scalar;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[path = "neon.rs"]
mod simd;

#[cfg(target_arch = "x86_64")]
#[path = "x86_64.rs"]
mod simd;

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
)))]
use scalar as simd;

/// A trait for determining splitpoints in a content-defined way.
pub trait Cdc {
    /// The amount of bytes needed before position `i` to determine if `i` is a
    /// splitpoint.
    ///
    /// Should return a constant.
    fn window_size(&self) -> usize;

    /// Returns the best splitpoint `i <= bytes.len()`, indicating bytes is to
    /// be split into `bytes[..i]` and `bytes[i..]`.
    fn best_splitpoint(&self, bytes: &[u8]) -> usize;
}

/// An instance of MinCDC4.
///
/// This chooses the first splitpoint `i` where
/// `u32::from_le_bytes(bytes[i-4..i])` is minimized.
#[non_exhaustive]
#[derive(Copy, Clone, Default, Debug)]
pub struct MinCdc4;

impl MinCdc4 {
    /// Create a new instance of MinCDC4.
    #[deprecated = "Unless you have a specific reason to use MinCdc4, prefer MinCdcHash4 instead. MinCdc4 is less robust to certain input patterns, and can easily create skewed chunk sizes. It is not recommended for general use, but is kept for academic purposes."]
    pub const fn new() -> Self {
        Self
    }
}

impl Cdc for MinCdc4 {
    #[inline(always)]
    fn window_size(&self) -> usize {
        4
    }

    #[inline(always)]
    fn best_splitpoint(&self, bytes: &[u8]) -> usize {
        if bytes.len() < 4 {
            return bytes.len();
        }
        4 + simd::argmin_u32_overlapping_hashed::<false>(bytes, 1, 0)
    }
}

/// An instance of MinCDCHash4.
///
/// This chooses the first splitpoint `i` where
/// `hash(u32::from_le_bytes(bytes[i-4..i]))` is minimized, where
/// `hash(v) = v.wrapping_mul(multiplier).wrapping_add(addend)`.
#[derive(Copy, Clone, Debug)]
pub struct MinCdcHash4 {
    multiplier: u32,
    addend: u32,
}

impl Default for MinCdcHash4 {
    fn default() -> Self {
        Self::new()
    }
}

impl MinCdcHash4 {
    /// Create a new instance of MinCDCHash4 with the default hash parameters.
    pub const fn new() -> Self {
        Self::with_params(DEFAULT_MULTIPLIER, DEFAULT_ADDEND)
    }

    /// Create a new instance of MinCDCHash4, specifying the hash parameters.
    ///
    /// # Panics
    /// Panics if the multiplier isn't odd. An even multiplier is always
    /// strictly worse.
    pub const fn with_params(multiplier: u32, addend: u32) -> Self {
        assert!(multiplier % 2 == 1, "the MinCDCHash multiplier must be odd");
        Self { multiplier, addend }
    }
}

impl Cdc for MinCdcHash4 {
    #[inline(always)]
    fn window_size(&self) -> usize {
        4
    }

    #[inline(always)]
    fn best_splitpoint(&self, bytes: &[u8]) -> usize {
        if bytes.len() < 4 {
            return bytes.len();
        }
        4 + simd::argmin_u32_overlapping_hashed::<true>(bytes, self.multiplier, self.addend)
    }
}

/// A chunker for a byte slice.
#[derive(Clone)]
pub struct SliceChunker<'a, C> {
    min_size: usize,
    max_size: usize,
    cdc: C,
    bytes: &'a [u8],
    offset: usize,
}

impl<'a, C> SliceChunker<'a, C> {
    /// Creates a new [`SliceChunker`] with the given minimum and maximum chunk
    /// size and CDC instance.
    ///
    /// The maximum size is always respected, however the final chunk may not
    /// respect the minimum size.
    ///
    /// # Panics
    /// Panics if `min_size > max_size` or `max_size == 0`.
    pub const fn new(bytes: &'a [u8], min_size: usize, max_size: usize, cdc: C) -> Self {
        assert!(min_size <= max_size && max_size > 0);

        Self {
            min_size,
            max_size,
            cdc,
            bytes,
            offset: 0,
        }
    }
}

/// The length of the next chunk at the front of the available bytes `avail`, or
/// `None` if it cannot be decided without more data (only possible when not at
/// end of input, i.e. `!eof`). This is the single source of truth for chunk
/// boundary placement, shared by [`SliceChunker`], [`ReadChunker`], and the
/// caterpillar layer so they cannot silently diverge.
pub(crate) fn next_chunk_len<C: Cdc>(
    avail: &[u8],
    min_size: usize,
    max_size: usize,
    eof: bool,
    cdc: &C,
) -> Option<usize> {
    let n = avail.len();
    if n == 0 {
        return None;
    }
    // Can't reliably place a boundary without the full decision window, unless
    // this is all the data there is.
    if !eof && n < max_size + 1 {
        return None;
    }
    if n <= min_size {
        return Some(n); // final short chunk (only reachable at eof)
    }
    let start = min_size.saturating_sub(cdc.window_size());
    Some(start + cdc.best_splitpoint(&avail[start..max_size.min(n)]))
}

impl<'a, C: Cdc> Iterator for SliceChunker<'a, C> {
    type Item = Chunk<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.bytes.len() {
            return None;
        }
        // The whole input is in memory, so we always know where the end is.
        let len = next_chunk_len(
            &self.bytes[self.offset..],
            self.min_size,
            self.max_size,
            true,
            &self.cdc,
        )
        .expect("eof=true always yields a chunk for non-empty input");
        let ret = Chunk::new(&self.bytes[self.offset..self.offset + len], self.offset);
        self.offset += len;
        Some(ret)
    }
}

impl<'a, C: Cdc> FusedIterator for SliceChunker<'a, C> {}

/// A chunker for a reader implementing [`Read`].
///
/// Note that unlike [`SliceChunker`] this stores bytes in an internal buffer
/// which is re-used and thus it can not implement [`Iterator`].
#[derive(Clone)]
pub struct ReadChunker<R, C> {
    min_size: usize,
    max_size: usize,
    cdc: C,
    reader: R,
    buf: Vec<u8>,
    buf_offset: usize,
    unread_bytes_in_buf: usize,
    stream_offset: usize,
}

impl<R, C: Cdc> ReadChunker<R, C> {
    /// Creates a new [`ReadChunker`] with the given minimum and maximum chunk
    /// size and CDC instance.
    ///
    /// The maximum size is always respected, however the final chunk may not
    /// respect the minimum size.
    ///
    /// # Panics
    /// Panics if `min_size > max_size` or `max_size == 0`.
    pub fn new(reader: R, min_size: usize, max_size: usize, cdc: C) -> Self {
        assert!(min_size <= max_size && max_size > 0);

        let bytes_needed_for_decision = max_size + 1;
        let buf_size = MIN_BUFFER_SIZE + bytes_needed_for_decision + min_size * 4;
        Self {
            min_size,
            max_size,
            cdc,
            reader,
            buf: vec![0; buf_size],
            buf_offset: 0,
            unread_bytes_in_buf: 0,
            stream_offset: 0,
        }
    }
}

impl<R: Read, C: Cdc> ReadChunker<R, C> {
    /// Gets the next [`Chunk`] from the reader, or [`None`] if it is exhausted.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> io::Result<Option<Chunk<'_>>> {
        if self.stream_offset == usize::MAX {
            return Ok(None);
        }

        let bytes_needed_for_decision = self.max_size + 1;
        while self.unread_bytes_in_buf < bytes_needed_for_decision {
            // We can't fit bytes_needed_for_decision anymore, we need to shift back.
            if self.buf.len() - self.buf_offset < bytes_needed_for_decision {
                self.buf.copy_within(
                    self.buf_offset..self.buf_offset + self.unread_bytes_in_buf,
                    0,
                );
                self.buf_offset = 0;
            }

            let bytes_read = self
                .reader
                .read(&mut self.buf[self.buf_offset + self.unread_bytes_in_buf..])?;
            if bytes_read == 0 {
                break;
            }
            self.unread_bytes_in_buf += bytes_read;
        }

        // The fill loop stops at `bytes_needed_for_decision` (= max+1) or EOF, so
        // if we have fewer than that, we've hit the end of the reader.
        let avail = &self.buf[self.buf_offset..self.buf_offset + self.unread_bytes_in_buf];
        let eof = self.unread_bytes_in_buf < bytes_needed_for_decision;
        match next_chunk_len(avail, self.min_size, self.max_size, eof, &self.cdc) {
            None => {
                // Reader is exhausted.
                self.stream_offset = usize::MAX;
                Ok(None)
            },
            Some(len) => {
                let ret = Chunk::new(
                    &self.buf[self.buf_offset..self.buf_offset + len],
                    self.stream_offset,
                );
                self.stream_offset += len;
                self.buf_offset += len;
                self.unread_bytes_in_buf -= len;
                Ok(Some(ret))
            },
        }
    }
}

/// A chunk returned by [`SliceChunker`] or [`ReadChunker`].
///
/// This implements [`Deref`] so you can directly treat it as a byte slice.
#[derive(Copy, Clone, Debug)]
pub struct Chunk<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Chunk<'a> {
    /// Creates a new [`Chunk`] with the given `bytes` and `offset`.
    pub const fn new(bytes: &'a [u8], offset: usize) -> Self {
        Self { bytes, offset }
    }

    /// The start offset of this chunk within the full data.
    pub const fn offset(&self) -> usize {
        self.offset
    }
}

impl<'a> Deref for Chunk<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.bytes
    }
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use rand::distr::StandardUniform;
    use rand::prelude::*;

    use crate::{
        DEFAULT_ADDEND, DEFAULT_MULTIPLIER, MinCdc4, MinCdcHash4, ReadChunker, SliceChunker,
        scalar, simd,
    };

    #[test]
    fn test_argmin_overlapped() {
        for size in 0..4096 {
            let rng = SmallRng::seed_from_u64(size);
            let bytes: Vec<u8> = rng
                .sample_iter(StandardUniform)
                .take(size as usize)
                .collect();
            assert_eq!(
                simd::argmin_u32_overlapping_hashed::<false>(&bytes, 1, 0),
                scalar::argmin_u32_overlapping_hashed::<false>(&bytes, 1, 0)
            );
            assert_eq!(
                simd::argmin_u32_overlapping_hashed::<true>(
                    &bytes,
                    DEFAULT_MULTIPLIER,
                    DEFAULT_ADDEND
                ),
                scalar::argmin_u32_overlapping_hashed::<true>(
                    &bytes,
                    DEFAULT_MULTIPLIER,
                    DEFAULT_ADDEND
                )
            );
        }
    }

    // The dispatched packed-scan primitives (whatever SIMD width this target
    // compiled in) must agree with the scalar reference for every size and
    // every mismatch position, including inside the sub-vector tail.
    #[test]
    fn test_packed_scan_matches_scalar() {
        for size in 0..600usize {
            let rng = SmallRng::seed_from_u64(size as u64);
            let a: Vec<u8> = rng.sample_iter(StandardUniform).take(size).collect();
            for p in 0..=size {
                let mut b = a.clone();
                if p < size {
                    b[p] ^= 0x5A;
                }
                let want = p.min(size);
                assert_eq!(scalar::common_prefix_len(&a, &b), want, "scalar {size}/{p}");
                assert_eq!(simd::common_prefix_len(&a, &b), want, "simd {size}/{p}");

                let mut run = vec![0x77u8; size];
                if p < size {
                    run[p] = 0;
                }
                assert_eq!(
                    scalar::byte_run_len(&run, 0x77),
                    want,
                    "scalar run {size}/{p}"
                );
                assert_eq!(simd::byte_run_len(&run, 0x77), want, "simd run {size}/{p}");
            }
        }
    }

    #[test]
    fn test_read_slice_equiv() {
        let bounds = [1, 2, 3, 4, 6, 8, 15, 27, 62, 90, 120, 200];
        for min_size in &bounds {
            for max_size in &bounds {
                if min_size > max_size {
                    continue;
                }
                for size in 0..4096 {
                    let rng = SmallRng::seed_from_u64(size);
                    let bytes: Vec<u8> = rng
                        .sample_iter(StandardUniform)
                        .take(size as usize)
                        .collect();

                    let reader = Cursor::new(&bytes);
                    let mut read_chunker = ReadChunker::new(reader, *min_size, *max_size, MinCdc4);
                    let slice_chunker = SliceChunker::new(&bytes, *min_size, *max_size, MinCdc4);
                    for slice_chunk in slice_chunker {
                        let read_chunk = read_chunker.next().unwrap().unwrap();
                        assert_eq!(slice_chunk.offset(), read_chunk.offset());
                        assert_eq!(&slice_chunk[..], &read_chunk[..]);
                    }
                    assert!(read_chunker.next().unwrap().is_none());

                    let reader = Cursor::new(&bytes);
                    let mut read_chunker =
                        ReadChunker::new(reader, *min_size, *max_size, MinCdcHash4::new());
                    let slice_chunker =
                        SliceChunker::new(&bytes, *min_size, *max_size, MinCdcHash4::new());
                    for slice_chunk in slice_chunker {
                        let read_chunk = read_chunker.next().unwrap().unwrap();
                        assert_eq!(slice_chunk.offset(), read_chunk.offset());
                        assert_eq!(&slice_chunk[..], &read_chunk[..]);
                    }
                    assert!(read_chunker.next().unwrap().is_none());
                }
            }
        }
    }
}
