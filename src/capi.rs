//! Minimal C API (feature `capi`) for embedding mothcdc in external
//! harnesses — written for the UWASL dedup-bench integration, where a C++
//! `Chunking_Technique` drives this library through one function. Not part
//! of the stable Rust API.

use crate::caterpillar;
use crate::mincdc::{MinCdcHash4, next_chunk_len};

/// Length of the next chunk at the front of `data[..len]`, using
/// [`MinCdcHash4`] with the default parameters.
///
/// `eof != 0` means `data` is all the data there is. With `eof == 0` at
/// least `max_size + 1` bytes must be available, or the boundary is
/// undecidable and 0 is returned (0 is also returned for `len == 0`).
///
/// If `repeats_out` is non-null it receives the number of *additional*
/// chunks — each byte-identical to this one and of the same length — that
/// are guaranteed to follow (the caterpillar packed fast path). The caller
/// may consume that many chunks of the returned length without calling back
/// in; the guarantee holds even if more stream data is buffered in between,
/// because those boundaries are pure functions of bytes already proven
/// periodic. Pass null to skip the packed scan (plain MinCDC).
///
/// # Safety
/// `data` must point to `len` readable bytes. `repeats_out` must be null or
/// point to a writable `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mothcdc_next_chunk(
    data: *const u8,
    len: usize,
    min_size: usize,
    max_size: usize,
    eof: core::ffi::c_int,
    repeats_out: *mut usize,
) -> usize {
    if !repeats_out.is_null() {
        unsafe { *repeats_out = 0 };
    }
    if data.is_null() || len == 0 || min_size > max_size || max_size == 0 {
        return 0;
    }
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    let cdc = MinCdcHash4::new();
    let Some(chunk_len) = next_chunk_len(bytes, min_size, max_size, eof != 0, &cdc) else {
        return 0;
    };
    if !repeats_out.is_null() {
        unsafe { *repeats_out = caterpillar::packed_repeats(bytes, chunk_len, max_size) };
    }
    chunk_len
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mincdc::SliceChunker;

    /// Driving the C API the way dedup-bench's chunk_stream does (a sliding
    /// buffer, consuming `repeats` boundaries without calling back in) must
    /// reproduce SliceChunker's boundaries exactly.
    #[test]
    fn capi_stream_protocol_matches_slice_chunker() {
        let (min, max) = (64usize, 256usize);
        let buf_cap = 1024usize; // like dedup-bench's buffer_size, > max + 1

        let mut data = Vec::new();
        // random | zero run | periodic run | random tail
        let mut s = 7u64;
        let mut rand = |n: usize| {
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s >> 33) as u8
                })
                .collect::<Vec<u8>>()
        };
        data.extend_from_slice(&rand(3000));
        data.extend_from_slice(&vec![0u8; 5000]);
        let unit = rand(37);
        for _ in 0..200 {
            data.extend_from_slice(&unit);
        }
        data.extend_from_slice(&rand(2000));

        let want: Vec<usize> = SliceChunker::new(&data, min, max, MinCdcHash4::new())
            .map(|c| c.len())
            .collect();

        // Emulate chunk_stream: sliding window over the data, repeats honored.
        let mut got = Vec::new();
        let mut pos = 0usize;
        let mut pending_repeats = 0usize;
        let mut pending_len = 0usize;
        while pos < data.len() {
            let end = (pos + buf_cap).min(data.len());
            let window = &data[pos..end];
            let len = if pending_repeats > 0 {
                pending_repeats -= 1;
                pending_len
            } else {
                let mut reps = 0usize;
                let eof = (window.len() < max + 1) as core::ffi::c_int;
                let l = unsafe {
                    mothcdc_next_chunk(window.as_ptr(), window.len(), min, max, eof, &mut reps)
                };
                assert!(l > 0, "boundary must be decidable under the eof rule");
                pending_repeats = reps;
                pending_len = l;
                l
            };
            got.push(len);
            pos += len;
        }
        assert_eq!(
            got, want,
            "C API stream protocol diverged from SliceChunker"
        );
    }
}
