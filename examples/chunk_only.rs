// Example program that only benchmarks chunking speed on a file read into memory.

use std::error::Error;
use std::fs::File;
use std::io::Read;

use mothcdc::{MinCdc4, MinCdcHash4, SliceChunker};

fn chunk_slice(bytes: &[u8], alg: &str, min_chunk_size: usize, mean_chunk_size: usize) {
    match alg {
        "mincdc4" => {
            #[allow(deprecated)]
            let cdc = MinCdc4::new();
            let max_chunk_size = mean_chunk_size + mean_chunk_size - min_chunk_size;
            let chunker = SliceChunker::new(bytes, min_chunk_size, max_chunk_size, cdc);
            for chunk in chunker {
                std::hint::black_box(chunk[0]);
            }
        },
        "mincdchash4" => {
            let cdc = MinCdcHash4::new();
            let max_chunk_size = mean_chunk_size + mean_chunk_size - min_chunk_size;
            let chunker = SliceChunker::new(bytes, min_chunk_size, max_chunk_size, cdc);
            for chunk in chunker {
                std::hint::black_box(chunk[0]);
            }
        },
        "fastcdc2020" => {
            // FastCDC wants assymmetric min/max sizes to not hit chunk limit too often.
            let max_chunk_size = mean_chunk_size + (mean_chunk_size - min_chunk_size) * 7;
            let chunker = fastcdc::v2020::FastCDC::new(
                bytes,
                min_chunk_size,
                mean_chunk_size,
                max_chunk_size,
            );
            for chunk in chunker {
                std::hint::black_box(bytes[chunk.offset]);
            }
        },
        _ => panic!("unknown algorithm: '{}'", alg),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let alg = &args[1];
    let min_chunk_size: usize = args[2].parse().unwrap();
    let mean_chunk_size: usize = args[3].parse().unwrap();
    let path = &args[4];

    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let start = std::time::Instant::now();
    for iter in 1.. {
        chunk_slice(&bytes, alg, min_chunk_size, mean_chunk_size);
        let mb_per_sec =
            (iter * bytes.len()) as f64 / start.elapsed().as_secs_f64() / (1000.0 * 1000.0);
        println!("{} ({:.2} MB / sec)", bytes.len(), mb_per_sec);
    }
    Ok(())
}
