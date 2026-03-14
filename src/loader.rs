use crate::model::Document;
use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;
use serde_yaml::Value;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

const SUPPORTED_EXT: [&str; 4] = ["md", "txt", "html", "json"];

pub fn load_documents(root: &Path) -> Result<Vec<Document>> {
    let mut docs = Vec::new();

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.into_path();
        if !is_supported(&path) {
            continue;
        }

        let raw = fs::read_to_string(&path)
            .with_context(|| format!("reading document {}", path.display()))?;
        let (metadata, text) = split_frontmatter(&raw)?;

        let id = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();

        let title = extract_title(&text);

        docs.push(Document {
            id,
            path: path.clone(),
            title,
            text,
            metadata,
        });
    }

    Ok(docs)
}

fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| SUPPORTED_EXT.contains(&ext))
        .unwrap_or(false)
}

fn extract_title(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            return Some(trimmed.trim_start_matches('#').trim().to_string());
        } else {
            return Some(trimmed.to_string());
        }
    }
    None
}

fn split_frontmatter(input: &str) -> Result<(std::collections::HashMap<String, String>, String)> {
    // basic YAML frontmatter: ---\n...\n---
    let re = Regex::new(r"(?s)^---\s*\n(.*?)\n---\s*\n?").unwrap();
    if let Some(caps) = re.captures(input) {
        let fm = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let body = &input[caps.get(0).unwrap().end()..];
        let map = parse_frontmatter(fm)?;
        Ok((map, body.to_string()))
    } else {
        Ok((Default::default(), input.to_string()))
    }
}

fn parse_frontmatter(fm: &str) -> Result<std::collections::HashMap<String, String>> {
    #[derive(Deserialize)]
    struct AnyMap(std::collections::HashMap<String, Value>);
    let parsed: AnyMap = serde_yaml::from_str(fm).context("parsing frontmatter yaml")?;
    let mut out = std::collections::HashMap::new();
    for (k, v) in parsed.0 {
        if let Some(s) = v.as_str() {
            out.insert(k, s.to_string());
        } else if let Some(n) = v.as_i64() {
            out.insert(k, n.to_string());
        } else if let Some(b) = v.as_bool() {
            out.insert(k, b.to_string());
        }
    }
    Ok(out)
}
