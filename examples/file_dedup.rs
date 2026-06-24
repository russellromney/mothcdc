// Example program that benchmarks chunking speed on files read from disk,
// including a cheap fingerprint hash to identify duplicate chunks. It outputs
// the speed per file, as well as statistics on the chunk size distribution and
// deduplication ratio.
//
// Usage:
//
//   cargo run -r --example file_dedup -- <algorithm> <min_chunk_size> <avg_chunk_size> <file1> <file2> ...
//
// where <algorithm> is one of "mincdc4", "mincdchash4", or "fastcdc2020".

use std::collections::HashMap;
use std::fs::File;
use std::hash::BuildHasher;
use std::io::{Result, Write};

use foldhash::quality::FixedState;
use mincatcdc::{MinCdc4, MinCdcHash4, ReadChunker};

fn chunk_file(
    digest_count: &mut HashMap<u64, usize>,
    file: File,
    alg: &str,
    min_chunk_size: usize,
    avg_chunk_size: usize,
) -> Result<()> {
    match alg {
        "mincdc4" => {
            #[allow(deprecated)]
            let cdc = MinCdc4::new();
            let max_chunk_size = avg_chunk_size + avg_chunk_size - min_chunk_size;
            let mut chunker = ReadChunker::new(file, min_chunk_size, max_chunk_size, cdc);
            while let Some(chunk) = chunker.next()? {
                let digest = FixedState::default().hash_one(&*chunk);
                digest_count.insert(digest, chunk.len());
            }
        },
        "mincdchash4" => {
            let cdc = MinCdcHash4::new();
            let max_chunk_size = avg_chunk_size + avg_chunk_size - min_chunk_size;
            let mut chunker = ReadChunker::new(file, min_chunk_size, max_chunk_size, cdc);
            while let Some(chunk) = chunker.next()? {
                let digest = FixedState::default().hash_one(&*chunk);
                digest_count.insert(digest, chunk.len());
            }
        },
        "fastcdc2020" => {
            // FastCDC wants assymmetric min/max sizes to not hit chunk limit too often.
            let max_chunk_size = avg_chunk_size + (avg_chunk_size - min_chunk_size) * 7;
            let mut chunker = fastcdc::v2020::StreamCDC::new(
                file,
                min_chunk_size,
                avg_chunk_size,
                max_chunk_size,
            );
            while let Some(Ok(chunk)) = chunker.next() {
                let digest = FixedState::default().hash_one(chunk.data);
                digest_count.insert(digest, chunk.length);
            }
        },
        _ => panic!("unknown algorithm: '{}'", alg),
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let alg = &args[1];
    let min_chunk_size: usize = args[2].parse().unwrap();
    let avg_chunk_size: usize = args[3].parse().unwrap();
    let paths = &args[4..];

    let mut digest_count: HashMap<u64, usize> = HashMap::new();
    let mut total_size = 0;
    for path in paths {
        let file = File::open(&path)?;
        let len = file.metadata()?.len();
        total_size += len;

        let start = std::time::Instant::now();
        chunk_file(&mut digest_count, file, alg, min_chunk_size, avg_chunk_size)?;
        let mb_per_sec = len as f64 / start.elapsed().as_secs_f64() / (1000.0 * 1000.0);
        println!("{path}: {} ({:.2} MB / sec)", len, mb_per_sec);
    }

    let mut sizes: Vec<_> = digest_count.values().copied().collect();
    if !sizes.is_empty() {
        sizes.sort();
        println!("min size: {}", sizes.first().unwrap());
        println!(" 1% size: {}", sizes[sizes.len() / 100]);
        println!("10% size: {}", sizes[sizes.len() / 10]);
        println!("25% size: {}", sizes[sizes.len() / 4]);
        println!("50% size: {}", sizes[sizes.len() / 2]);
        println!(
            "75% size: {}",
            sizes[(sizes.len() - 1).saturating_sub(sizes.len() / 4)]
        );
        println!(
            "90% size: {}",
            sizes[(sizes.len() - 1).saturating_sub(sizes.len() / 10)]
        );
        println!(
            "99% size: {}",
            sizes[(sizes.len() - 1).saturating_sub(sizes.len() / 100)]
        );
        println!("max size: {}", sizes.last().unwrap());
        println!(
            "mean size: {:.1}",
            (sizes.iter().sum::<usize>() as f64 / sizes.len() as f64)
        );

        if let Ok(fname) = std::env::var("OUTPUT_SIZES") {
            let mut file = File::create(fname)?;
            for size in sizes {
                writeln!(file, "{size}")?;
            }
        }
    }

    let dedup_size = digest_count.values().sum::<usize>();
    let fraction = 100.0 - dedup_size as f64 / total_size as f64 * 100.0;
    println!("{} unique blocks", digest_count.len());
    println!("{dedup_size} dedup size ({fraction:.3}% savings)");

    Ok(())
}
