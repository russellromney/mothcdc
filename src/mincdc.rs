//! MinCDC
//! ------
//! This module inherits the core chunking algorithm from Orson Peters'
//! [MinCDC](https://github.com/orlp/mincdc): a very simple yet efficient
//! content-defined chunking algorithm. This crate provides a SIMD-accelerated
//! implementation of it.
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
//!  - [`MinCdc4`](crate::mincdc::MinCdc4), where the evaluation function is
//!    `u32::from_le_bytes(bytes[i - 4..i])`, i.e. a window size of 4 bytes
//!    interpreting the bytes as a little-endian `u32`, and
//!  - [`MinCdcHash4`](crate::mincdc::MinCdcHash4), where the evaluation function is
//!    `hash(u32::from_le_bytes(bytes[i - 4..i]))`. The hash function used is
//!    the very simple `hash(x) = x.wrapping_mul(a).wrapping_add(b)`, for
//!    some constants `a` and `b`.
//!
//! **[`MinCdcHash4`](crate::mincdc::MinCdcHash4) can be slightly (~10%) slower but is far more robust and
//! predictable, it is the recommended default**.
//!
//! # Usage
//!
//! This module provides two chunkers:
//!  - [`SliceChunker`](crate::mincdc::SliceChunker) for chunking a byte slice, and
//!  - [`ReadChunker`](crate::mincdc::ReadChunker) for chunking a reader implementing
//!    [`Read`](std::io::Read).
//!
//! Both chunkers take a desired minimum and maximum chunk size as well as a
//! [`Cdc`](crate::mincdc::Cdc) instance (either
//! [`MinCdc4`](crate::mincdc::MinCdc4) or
//! [`MinCdcHash4`](crate::mincdc::MinCdcHash4)). Then by iterating over the chunker
//! (or calling [`next()`](crate::mincdc::ReadChunker::next) in the case of
//! [`ReadChunker`](crate::mincdc::ReadChunker)) you get chunks of type
//! [`Chunk`], which derefs to a byte slice, but also contains the offset of
//! that chunk in the input stream.
//!
//! # Examples
//!
//! ```rust
//! # use mothcdc::mincdc::{MinCdcHash4, SliceChunker};
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

use std::io::{self, Read};
use std::iter::FusedIterator;
use std::ops::Deref;

use crate::MIN_BUFFER_SIZE;
use crate::simd;

pub(crate) const DEFAULT_MULTIPLIER: u32 = 0x915f77f5;
pub(crate) const DEFAULT_ADDEND: u32 = 0x34636463;

/// Largest supported boundary-search window.
///
/// SIMD implementations store candidate offsets in `u32` lanes. Keeping this
/// limit consistent across targets also keeps chunk boundaries portable.
pub const MAX_CHUNK_SIZE: usize = u32::MAX as usize - 1;

/// A trait for determining splitpoints in a content-defined way.
pub trait Cdc {
    /// The amount of bytes needed before position `i` to determine if `i` is a
    /// splitpoint.
    ///
    /// Must return the same non-zero value for the lifetime of the chunker.
    fn window_size(&self) -> usize;

    /// Returns the best splitpoint `i`, indicating bytes is to be split into
    /// `bytes[..i]` and `bytes[i..]`.
    ///
    /// # Contract
    ///
    /// If `bytes.len() < self.window_size()`, this must return `bytes.len()`.
    /// Otherwise it must return a value in
    /// `self.window_size()..=bytes.len()`. Chunkers enforce this contract.
    fn best_splitpoint(&self, bytes: &[u8]) -> usize;
}

/// An instance of MinCDC4.
///
/// This chooses the first splitpoint `i` where
/// `u32::from_le_bytes(bytes[i-4..i])` is minimized.
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
    pub(crate) max_size: usize,
    cdc: C,
    bytes: &'a [u8],
    pub(crate) offset: usize,
}

impl<'a, C> SliceChunker<'a, C> {
    /// Creates a new [`SliceChunker`] with the given minimum and maximum chunk
    /// size and CDC instance.
    ///
    /// The maximum size is always respected, however the final chunk may not
    /// respect the minimum size.
    ///
    /// # Panics
    /// Panics if the sizes are invalid or `max_size` exceeds
    /// [`MAX_CHUNK_SIZE`].
    pub const fn new(bytes: &'a [u8], min_size: usize, max_size: usize, cdc: C) -> Self {
        assert!(min_size <= max_size && max_size > 0 && max_size <= MAX_CHUNK_SIZE);

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
    if !eof && n <= max_size {
        return None;
    }
    if n <= min_size {
        return Some(n); // final short chunk (only reachable at eof)
    }
    let window = cdc.window_size();
    assert!(window > 0, "Cdc::window_size() must be non-zero");
    let start = min_size.saturating_sub(window);
    let search = &avail[start..max_size.min(n)];
    let split = cdc.best_splitpoint(search);
    let minimum = window.min(search.len());
    assert!(
        split >= minimum && split <= search.len(),
        "Cdc::best_splitpoint() returned {split} for {} bytes with window size {window}; expected {minimum}..={}",
        search.len(),
        search.len()
    );
    Some(start + split)
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
        let ret = Chunk::new(
            &self.bytes[self.offset..self.offset + len],
            self.offset as u64,
        );
        self.offset += len;
        Some(ret)
    }
}

impl<'a, C: Cdc> FusedIterator for SliceChunker<'a, C> {}

/// A chunker for a reader implementing [`Read`].
///
/// Note that unlike [`SliceChunker`] this stores bytes in an internal buffer
/// which is re-used and thus it can not implement [`Iterator`]. It allocates at
/// least 4 MiB so that ordinary reads amortize buffer movement.
#[derive(Clone)]
pub struct ReadChunker<R, C> {
    min_size: usize,
    max_size: usize,
    cdc: C,
    reader: R,
    buf: Vec<u8>,
    buf_offset: usize,
    unread_bytes_in_buf: usize,
    stream_offset: u64,
    done: bool,
}

impl<R, C: Cdc> ReadChunker<R, C> {
    /// Creates a new [`ReadChunker`] with the given minimum and maximum chunk
    /// size and CDC instance.
    ///
    /// The maximum size is always respected, however the final chunk may not
    /// respect the minimum size.
    ///
    /// # Panics
    /// Panics if the sizes are invalid, arithmetic overflows, or the internal
    /// buffer cannot be allocated. Use [`ReadChunker::try_new`] to handle those
    /// cases as errors.
    pub fn new(reader: R, min_size: usize, max_size: usize, cdc: C) -> Self {
        Self::try_new(reader, min_size, max_size, cdc)
            .expect("invalid ReadChunker configuration or buffer allocation failed")
    }

    /// Tries to create a reader chunker without arithmetic or allocation panics.
    /// Invalid sizes produce [`io::ErrorKind::InvalidInput`]; allocation failure
    /// produces [`io::ErrorKind::OutOfMemory`].
    pub fn try_new(reader: R, min_size: usize, max_size: usize, cdc: C) -> io::Result<Self> {
        let buf_size = checked_buffer_size(min_size, max_size)?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(buf_size)
            .map_err(|e| io::Error::new(io::ErrorKind::OutOfMemory, e))?;
        buf.resize(buf_size, 0);
        Ok(Self {
            min_size,
            max_size,
            cdc,
            reader,
            buf,
            buf_offset: 0,
            unread_bytes_in_buf: 0,
            stream_offset: 0,
            done: false,
        })
    }

    /// Returns a shared reference to the underlying reader.
    pub fn get_ref(&self) -> &R {
        &self.reader
    }

    /// Returns a mutable reference to the underlying reader.
    ///
    /// Reading from it directly can invalidate chunker state.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Returns the underlying reader, discarding any bytes already read ahead.
    pub fn into_inner(self) -> R {
        self.reader
    }
}

pub(crate) fn checked_buffer_size(min_size: usize, max_size: usize) -> io::Result<usize> {
    if min_size > max_size || max_size == 0 || max_size > MAX_CHUNK_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected 0 <= min_size <= max_size <= {MAX_CHUNK_SIZE}, with max_size > 0"),
        ));
    }
    let decision = max_size
        .checked_add(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "max_size + 1 overflowed"))?;
    let slack = min_size
        .checked_mul(4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "min_size * 4 overflowed"))?;
    MIN_BUFFER_SIZE
        .checked_add(decision)
        .and_then(|n| n.checked_add(slack))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "buffer size overflowed"))
}

impl<R: Read, C: Cdc> ReadChunker<R, C> {
    /// Gets the next [`Chunk`] from the reader, or [`None`] if it is exhausted.
    ///
    /// A `read` returning `Ok(0)` is treated as end of input.
    /// [`io::ErrorKind::Interrupted`] is retried internally; any other error is
    /// returned, and progress already made is preserved so the next call can
    /// resume.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> io::Result<Option<Chunk<'_>>> {
        if self.done {
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

            let bytes_read = loop {
                match self
                    .reader
                    .read(&mut self.buf[self.buf_offset + self.unread_bytes_in_buf..])
                {
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    result => break result?,
                }
            };
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
                self.done = true;
                Ok(None)
            },
            Some(len) => {
                let ret = Chunk::new(
                    &self.buf[self.buf_offset..self.buf_offset + len],
                    self.stream_offset,
                );
                self.stream_offset =
                    self.stream_offset.checked_add(len as u64).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "stream offset exceeded u64::MAX",
                        )
                    })?;
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
    offset: u64,
}

impl<'a> Chunk<'a> {
    /// Creates a new [`Chunk`] with the given `bytes` and `offset`.
    pub const fn new(bytes: &'a [u8], offset: u64) -> Self {
        Self { bytes, offset }
    }

    /// The start offset of this chunk within the full data. Stream offsets use
    /// `u64` even on 32-bit targets.
    pub const fn offset(&self) -> u64 {
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
    use std::io::{self, Cursor, Read};

    use rand::distr::StandardUniform;
    use rand::prelude::*;

    use crate::mincdc::{
        DEFAULT_ADDEND, DEFAULT_MULTIPLIER, MinCdc4, MinCdcHash4, ReadChunker, SliceChunker,
    };
    use crate::{scalar, simd};

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
        // The integration invariant suite exercises a much larger matrix
        // against an independent oracle. Keep this unit smoke test small so a
        // normal debug `cargo test` remains fast.
        let bounds = [1, 4, 27, 200];
        for min_size in &bounds {
            for max_size in &bounds {
                if min_size > max_size {
                    continue;
                }
                for size in [0usize, 1, 3, 4, 17, 200, 511, 4096] {
                    let rng = SmallRng::seed_from_u64(size as u64);
                    let bytes: Vec<u8> = rng.sample_iter(StandardUniform).take(size).collect();

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

    #[derive(Clone, Copy)]
    struct InvalidCdc {
        window: usize,
        split: usize,
    }

    impl super::Cdc for InvalidCdc {
        fn window_size(&self) -> usize {
            self.window
        }

        fn best_splitpoint(&self, _bytes: &[u8]) -> usize {
            self.split
        }
    }

    #[test]
    #[should_panic(expected = "best_splitpoint")]
    fn invalid_cdc_cannot_yield_empty_chunks() {
        let mut chunks = SliceChunker::new(
            b"enough input to require a split",
            1,
            8,
            InvalidCdc {
                window: 1,
                split: 0,
            },
        );
        let _ = chunks.next();
    }

    #[test]
    #[should_panic(expected = "window_size")]
    fn invalid_cdc_window_is_rejected() {
        let mut chunks = SliceChunker::new(
            b"enough input",
            1,
            8,
            InvalidCdc {
                window: 0,
                split: 1,
            },
        );
        let _ = chunks.next();
    }

    struct InterruptOnce<R> {
        inner: R,
        interrupted: bool,
    }

    impl<R: Read> Read for InterruptOnce<R> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(io::ErrorKind::Interrupted.into());
            }
            self.inner.read(buf)
        }
    }

    #[test]
    fn read_chunker_retries_interrupted() {
        let data = vec![7u8; 1024];
        let reader = InterruptOnce {
            inner: Cursor::new(&data),
            interrupted: false,
        };
        let mut chunks = ReadChunker::new(reader, 16, 64, MinCdcHash4::new());
        let mut rebuilt = Vec::new();
        while let Some(chunk) = chunks.next().unwrap() {
            rebuilt.extend_from_slice(&chunk);
        }
        assert_eq!(rebuilt, data);
    }

    struct OneByteReader<R>(R);

    impl<R: Read> Read for OneByteReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(1);
            self.0.read(&mut buf[..n])
        }
    }

    #[test]
    fn read_chunker_handles_one_byte_reads() {
        let data: Vec<u8> = (0..8192).map(|i| (i * 31) as u8).collect();
        let want: Vec<_> = SliceChunker::new(&data, 64, 256, MinCdcHash4::new())
            .map(|c| (c.offset(), c.to_vec()))
            .collect();
        let mut chunker = ReadChunker::new(
            OneByteReader(Cursor::new(&data)),
            64,
            256,
            MinCdcHash4::new(),
        );
        let mut got = Vec::new();
        while let Some(chunk) = chunker.next().unwrap() {
            got.push((chunk.offset(), chunk.to_vec()));
        }
        assert_eq!(got, want);
    }

    struct ErrorAfterFirstRead {
        data: Vec<u8>,
        first: bool,
    }

    impl Read for ErrorAfterFirstRead {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.first {
                return Err(io::Error::other("reader failed"));
            }
            self.first = true;
            let n = self.data.len().min(buf.len());
            buf[..n].copy_from_slice(&self.data[..n]);
            Ok(n)
        }
    }

    #[test]
    fn read_chunker_preserves_progress_before_error() {
        let reader = ErrorAfterFirstRead {
            data: vec![0; 17],
            first: false,
        };
        let mut chunks = ReadChunker::new(reader, 8, 16, MinCdcHash4::new());
        assert!(chunks.next().unwrap().is_some());
        assert_eq!(chunks.next().unwrap_err().kind(), io::ErrorKind::Other);
    }

    #[test]
    fn fallible_constructor_rejects_overflowing_configuration() {
        let result = ReadChunker::try_new(
            Cursor::new(Vec::<u8>::new()),
            0,
            usize::MAX,
            MinCdcHash4::new(),
        );
        match result {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidInput),
            Ok(_) => panic!("overflowing configuration was accepted"),
        }
    }
}
