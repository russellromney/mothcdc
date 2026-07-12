//! MothCDC
//! -------
//! A fork of Orson Peters' [MinCDC](https://github.com/orlp/mincdc) that adds
//! a *caterpillar* coalescing layer for metadata-efficient content-defined
//! chunking on redundant data.
//!
//! The core chunking algorithm lives in [`mincdc`] (inherited from upstream
//! MinCDC); the coalescing layer lives in [`caterpillar`], built on top of it.
//!
//! The default [`MothChunker`] and [`MothReadChunker`] use the robust hashed
//! MinCDC boundary selector and emit validated [`Segment`] values. A segment is
//! either one ordinary chunk or a maximal run of identical adjacent chunks;
//! use [`Segment::dedup_key`] for storage and [`Segment::len`] for its logical
//! byte length.
//!
//! # In-memory example
//!
//! ```rust
//! use mothcdc::MothChunker;
//!
//! let data = vec![0u8; 64 * 4096]; // a long zero run
//! let segs: Vec<_> = MothChunker::new(&data, 4096, 12288).collect();
//!
//! // The whole zero run collapses into one record instead of ~64 chunks.
//! assert_eq!(segs.len(), 1);
//! assert!(segs[0].is_caterpillar());
//! assert_eq!(segs[0].len(), data.len() as u64);
//! ```
//!
//! # Streaming example
//!
//! ```rust
//! use std::io::Cursor;
//! use mothcdc::MothReadChunker;
//!
//! let data = vec![0u8; 128 * 4096];
//! let mut chunker = MothReadChunker::new(Cursor::new(&data), 4096, 12288);
//! let mut restored = Vec::new();
//! while let Some(segment) = chunker.next()? {
//!     // Streaming segments borrow the internal buffer until the next call.
//!     segment.reconstruct_into(&mut restored);
//! }
//! assert_eq!(restored, data);
//! # Ok::<(), std::io::Error>(())
//! ```

#![warn(missing_docs)]

pub(crate) const MIN_BUFFER_SIZE: usize = 1024 * 1024 * 4;

/// The core MinCDC chunking algorithm, inherited from Orson Peters'
/// [MinCDC](https://github.com/orlp/mincdc): [`mincdc::Cdc`], [`mincdc::MinCdc4`],
/// [`mincdc::MinCdcHash4`], [`mincdc::SliceChunker`], [`mincdc::ReadChunker`],
/// and [`mincdc::Chunk`].
pub mod mincdc;
pub use mincdc::Chunk;

/// Caterpillar coalescing layer: metadata efficiency on redundant data.
pub mod caterpillar;
pub use caterpillar::{MothChunker, MothReadChunker, Segment};

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
