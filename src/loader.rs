use crate::model::Document;
use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;
use serde_yaml::Value;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use walkdir::WalkDir;

const SUPPORTED_EXT: [&str; 4] = ["md", "txt", "html", "json"];

pub fn load_documents(root: &Path) -> Result<Vec<Document>> {
    if !root.exists() {
        anyhow::bail!("documents path does not exist: {}", root.display());
    }
    if !root.is_dir() {
        anyhow::bail!("documents path is not a directory: {}", root.display());
    }

    let mut docs = Vec::new();
    let mut paths = Vec::new();

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.into_path();
        if !is_supported(&path) {
            continue;
        }
        paths.push(path);
    }

    // Deterministic ordering keeps chunk IDs and diagnostics stable across runs.
    paths.sort_by(|a, b| {
        let ra = a.strip_prefix(root).unwrap_or(a).to_string_lossy();
        let rb = b.strip_prefix(root).unwrap_or(b).to_string_lossy();
        ra.cmp(&rb)
    });

    for path in paths {
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

    if docs.is_empty() {
        anyhow::bail!(
            "no supported documents found under {} (supported: .md, .txt, .html, .json)",
            root.display()
        );
    }

    Ok(docs)
}

fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| SUPPORTED_EXT.contains(&ext.to_ascii_lowercase().as_str()))
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
    if let Some(caps) = frontmatter_regex().captures(input) {
        let fm = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let body_start = caps.get(0).map(|m| m.end()).unwrap_or(0);
        let body = &input[body_start..];
        let map = parse_frontmatter(fm);
        Ok((map, body.to_string()))
    } else {
        Ok((Default::default(), input.to_string()))
    }
}

fn parse_frontmatter(fm: &str) -> std::collections::HashMap<String, String> {
    #[derive(Deserialize)]
    struct AnyMap(std::collections::HashMap<String, Value>);
    let parsed: AnyMap = match serde_yaml::from_str(fm) {
        Ok(v) => v,
        Err(_) => return std::collections::HashMap::new(),
    };
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
    out
}

fn frontmatter_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?").unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn loader_orders_documents_deterministically() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rag_audit_loader_{stamp}"));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("b.md"), "# B\nbody").unwrap();
        fs::write(root.join("a.md"), "# A\nbody").unwrap();

        let docs = load_documents(&root).unwrap();
        let ids: Vec<String> = docs.into_iter().map(|d| d.id).collect();
        assert_eq!(ids, vec!["a.md".to_string(), "b.md".to_string()]);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn parses_crlf_frontmatter() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rag_audit_loader_crlf_{stamp}"));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("doc.md"),
            "---\r\nproduct: payments\r\n---\r\n# Title\r\nbody\r\n",
        )
        .unwrap();

        let docs = load_documents(&root).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata.get("product"),
            Some(&"payments".to_string())
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invalid_frontmatter_is_tolerated() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rag_audit_loader_badfm_{stamp}"));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("doc.md"), "---\n: bad yaml\n---\n# Title\nbody\n").unwrap();

        let docs = load_documents(&root).unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].metadata.is_empty());
        assert!(docs[0].text.contains("# Title"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_missing_root_path() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rag_audit_loader_missing_{stamp}"));
        let err = load_documents(&root).expect_err("missing path should fail");
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn supports_uppercase_extensions() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rag_audit_loader_upper_{stamp}"));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("doc.MD"), "# Title\nbody").unwrap();

        let docs = load_documents(&root).expect("uppercase extension should be supported");
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "doc.MD");

        let _ = fs::remove_dir_all(root);
    }
}
