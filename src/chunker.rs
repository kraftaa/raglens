use crate::config::Config;
use crate::model::{Chunk, Document};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn chunk_documents(docs: &[Document], config: &Config) -> Vec<Chunk> {
    let counter = AtomicUsize::new(0);
    docs.par_iter()
        .flat_map(|doc| chunk_single(doc, config, &counter))
        .collect()
}

fn chunk_single(
    doc: &Document,
    config: &Config,
    counter: &AtomicUsize,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut buffer = String::new();
    let mut token_count = 0usize;
    let mut heading_path: Vec<String> = Vec::new();

    for line in doc.text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            heading_path = trimmed
                .trim_start_matches('#')
                .trim()
                .split('/')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        let line_tokens = token_estimate(trimmed);
        if token_count + line_tokens > config.chunk_size && !buffer.is_empty() {
            chunks.push(make_chunk(
                doc,
                buffer.trim(),
                token_count,
                heading_path.clone(),
                counter,
            ));
            // overlap by reusing tail tokens if requested
            if config.chunk_overlap > 0 {
                buffer = retain_tail(buffer, config.chunk_overlap);
                token_count = token_estimate(&buffer);
            } else {
                buffer.clear();
                token_count = 0;
            }
        }

        if !trimmed.is_empty() {
            buffer.push_str(trimmed);
            buffer.push('\n');
            token_count += line_tokens;
        }
    }

    if !buffer.trim().is_empty() {
        chunks.push(make_chunk(
            doc,
            buffer.trim(),
            token_count,
            heading_path,
            counter,
        ));
    }

    chunks
}

fn make_chunk(
    doc: &Document,
    text: &str,
    token_count: usize,
    heading_path: Vec<String>,
    counter: &AtomicUsize,
) -> Chunk {
    let idx = counter.fetch_add(1, Ordering::Relaxed);
    Chunk {
        chunk_id: format!("{}#{}", doc.id, idx),
        doc_id: doc.id.clone(),
        text: text.to_string(),
        token_count,
        heading_path,
    }
}

fn retain_tail(text: String, overlap_tokens: usize) -> String {
    let mut tokens = 0usize;
    let mut selected = Vec::new();
    for line in text.lines().rev() {
        let line_tokens = token_estimate(line);
        if tokens + line_tokens > overlap_tokens && tokens > 0 {
            break;
        }
        selected.push(line);
        tokens += line_tokens;
    }
    selected.reverse();
    selected.join("\n") + "\n"
}

fn token_estimate(text: &str) -> usize {
    text.split_whitespace().count()
}
