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
        if trimmed.is_empty() {
            continue;
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

        let segments = if trimmed.starts_with('#') {
            vec![trimmed.to_string()]
        } else {
            split_text_for_chunking(trimmed, config.chunk_size)
        };

        for segment in segments {
            let seg_tokens = token_estimate(&segment);
            if token_count + seg_tokens > config.chunk_size && !buffer.is_empty() {
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
            buffer.push_str(&segment);
            buffer.push('\n');
            token_count += seg_tokens;
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

fn split_text_for_chunking(text: &str, max_tokens: usize) -> Vec<String> {
    if max_tokens == 0 || token_estimate(text) <= max_tokens {
        return vec![text.to_string()];
    }

    let sentences = split_sentences(text);
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut buf_tokens = 0usize;

    for sentence in sentences {
        let sentence_tokens = token_estimate(&sentence);
        if sentence_tokens > max_tokens {
            if !buf.trim().is_empty() {
                out.push(buf.trim().to_string());
                buf.clear();
                buf_tokens = 0;
            }
            out.extend(split_by_tokens(&sentence, max_tokens));
            continue;
        }
        if buf_tokens + sentence_tokens > max_tokens && !buf.trim().is_empty() {
            out.push(buf.trim().to_string());
            buf.clear();
            buf_tokens = 0;
        }
        if !buf.is_empty() {
            buf.push(' ');
        }
        buf.push_str(&sentence);
        buf_tokens += sentence_tokens;
    }

    if !buf.trim().is_empty() {
        out.push(buf.trim().to_string());
    }
    out
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0usize;
    for (idx, ch) in text.char_indices() {
        if !matches!(ch, '.' | '!' | '?' | ';') {
            continue;
        }
        let next_start = idx + ch.len_utf8();
        let next = text.get(next_start..).and_then(|s| s.chars().next());
        if next.map(|c| c.is_whitespace()).unwrap_or(true) {
            let part = text[start..next_start].trim();
            if !part.is_empty() {
                out.push(part.to_string());
            }
            start = next_start;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    if out.is_empty() {
        vec![text.to_string()]
    } else {
        out
    }
}

fn split_by_tokens(text: &str, max_tokens: usize) -> Vec<String> {
    if max_tokens == 0 {
        return vec![text.to_string()];
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![text.to_string()];
    }
    words.chunks(max_tokens).map(|w| w.join(" ")).collect()
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

    #[test]
    fn long_single_line_is_split_by_token_limit() {
        let long_line = (0..120)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let doc = Document {
            id: "doc.md".to_string(),
            path: PathBuf::from("doc.md"),
            title: Some("Doc".to_string()),
            text: long_line,
            metadata: HashMap::new(),
        };
        let cfg = Config {
            chunk_size: 20,
            chunk_overlap: 0,
            ..Config::default()
        };
        let chunks = chunk_documents(&[doc], &cfg);
        assert!(chunks.len() > 1, "expected long single-line split");
        assert!(chunks.iter().all(|c| c.token_count <= 20));
    }

    #[test]
    fn sentence_split_keeps_sentence_boundaries_when_possible() {
        let doc = Document {
            id: "doc.md".to_string(),
            path: PathBuf::from("doc.md"),
            title: Some("Doc".to_string()),
            text: "First sentence has six words here. Second sentence also has six words."
                .to_string(),
            metadata: HashMap::new(),
        };
        let cfg = Config {
            chunk_size: 7,
            chunk_overlap: 0,
            ..Config::default()
        };
        let chunks = chunk_documents(&[doc], &cfg);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].text.ends_with('.'));
    }
}
