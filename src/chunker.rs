use crate::config::Config;
use crate::model::{Chunk, Document};

pub fn chunk_documents(docs: &[Document], config: &Config) -> Vec<Chunk> {
    let mut counter = 0usize;
    docs.iter()
        .flat_map(|doc| chunk_single(doc, config, &mut counter))
        .collect()
}

fn chunk_single(doc: &Document, config: &Config, counter: &mut usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut buffer = String::new();
    let mut token_count = 0usize;
    let mut heading_path: Vec<String> = Vec::new();

    for line in doc.text.lines() {
        let trimmed = line.trim();
        let line_tokens = token_estimate(trimmed);
        if token_count + line_tokens > config.chunk_size && !buffer.is_empty() {
            flush_token_split(
                &mut chunks,
                &mut buffer,
                &mut token_count,
                heading_path.clone(),
                doc,
                config,
                counter,
            );
        }

        // Split by heading boundaries first to keep chunks coherent per section.
        if trimmed.starts_with('#') && !buffer.trim().is_empty() {
            chunks.push(make_chunk(
                doc,
                buffer.trim(),
                token_count,
                heading_path.clone(),
                counter,
            ));
            buffer.clear();
            token_count = 0;
        }
        if trimmed.starts_with('#') {
            heading_path = parse_heading_path(trimmed);
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
    counter: &mut usize,
) -> Chunk {
    let idx = *counter;
    *counter += 1;
    Chunk {
        chunk_id: format!("{}#{}", doc.id, idx),
        doc_id: doc.id.clone(),
        text: text.to_string(),
        token_count,
        heading_path,
    }
}

fn flush_token_split(
    chunks: &mut Vec<Chunk>,
    buffer: &mut String,
    token_count: &mut usize,
    heading_path: Vec<String>,
    doc: &Document,
    config: &Config,
    counter: &mut usize,
) {
    chunks.push(make_chunk(
        doc,
        buffer.trim(),
        *token_count,
        heading_path,
        counter,
    ));
    // overlap by reusing tail tokens if requested
    if config.chunk_overlap > 0 {
        let retained = retain_tail(std::mem::take(buffer), config.chunk_overlap);
        *buffer = retained;
        *token_count = token_estimate(buffer);
    } else {
        buffer.clear();
        *token_count = 0;
    }
}

fn parse_heading_path(trimmed_heading: &str) -> Vec<String> {
    trimmed_heading
        .trim_start_matches('#')
        .trim()
        .split('/')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Document;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn heading_boundary_keeps_previous_chunk_heading() {
        let doc = Document {
            id: "doc.md".to_string(),
            path: PathBuf::from("doc.md"),
            title: Some("Doc".to_string()),
            text: "# Refund\nrefund line one\nrefund line two\n# Shipping\nshipping content\n"
                .to_string(),
            metadata: HashMap::new(),
        };
        let cfg = Config {
            chunk_size: 5,
            chunk_overlap: 0,
            ..Config::default()
        };
        let chunks = chunk_documents(&[doc], &cfg);
        assert!(chunks.len() >= 2, "expected split into at least two chunks");
        let shipping_chunk = chunks
            .iter()
            .find(|c| c.text.contains("shipping content"))
            .expect("missing shipping chunk");
        assert_eq!(shipping_chunk.heading_path, vec!["Shipping".to_string()]);
    }
}
