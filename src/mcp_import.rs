use crate::model::{RetrievedDoc, RunArtifact, RunContext};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const DEFAULT_QUESTION_POINTERS: &[&str] = &[
    "/question",
    "/query",
    "/input/question",
    "/request/question",
    "/request/query",
];

const DEFAULT_ANSWER_POINTERS: &[&str] = &[
    "/answer",
    "/final_answer",
    "/response/answer",
    "/response/output_text",
    "/output/answer",
    "/output_text",
    "/completion",
];

const DEFAULT_DOCS_POINTERS: &[&str] = &[
    "/retrieved_docs",
    "/retrieval/retrieved_docs",
    "/retrieval/docs",
    "/context/retrieved_docs",
    "/context/docs",
    "/docs",
    "/documents",
    "/chunks",
    "/passages",
];

pub struct McpImportOpts<'a> {
    pub question_pointer: Option<&'a str>,
    pub answer_pointer: Option<&'a str>,
    pub docs_pointer: Option<&'a str>,
    pub model: Option<String>,
    pub top_k: Option<usize>,
}

pub fn import_file(path: &Path, opts: &McpImportOpts<'_>) -> Result<RunArtifact> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let root: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing JSON {}", path.display()))?;
    import_value(&root, opts)
}

pub fn import_value(root: &Value, opts: &McpImportOpts<'_>) -> Result<RunArtifact> {
    let question = extract_string_field(
        root,
        opts.question_pointer,
        DEFAULT_QUESTION_POINTERS,
        "question",
    )?;
    let answer =
        extract_string_field(root, opts.answer_pointer, DEFAULT_ANSWER_POINTERS, "answer")?;
    let docs = extract_docs(root, opts.docs_pointer)?;

    Ok(RunArtifact {
        question,
        answer,
        retrieved_docs: docs,
        claims: Vec::new(),
        metrics: None,
        context: if opts.model.is_some() || opts.top_k.is_some() {
            Some(RunContext {
                model: opts.model.clone(),
                top_k: opts.top_k,
            })
        } else {
            None
        },
    })
}

fn extract_string_field(
    root: &Value,
    explicit_pointer: Option<&str>,
    default_pointers: &[&str],
    field_name: &str,
) -> Result<String> {
    if let Some(pointer) = explicit_pointer {
        let value = root
            .pointer(pointer)
            .ok_or_else(|| anyhow::anyhow!("{} pointer not found: {}", field_name, pointer))?;
        return value_to_string(value).ok_or_else(|| {
            anyhow::anyhow!(
                "{} pointer {} did not resolve to a string-like value",
                field_name,
                pointer
            )
        });
    }

    for pointer in default_pointers {
        if let Some(value) = root.pointer(pointer) {
            if let Some(s) = value_to_string(value) {
                if !s.trim().is_empty() {
                    return Ok(s);
                }
            }
        }
    }

    anyhow::bail!(
        "could not infer '{}' from input JSON; pass --{}-pointer",
        field_name,
        field_name
    )
}

fn extract_docs(root: &Value, explicit_pointer: Option<&str>) -> Result<Vec<RetrievedDoc>> {
    let docs_array = if let Some(pointer) = explicit_pointer {
        let value = root
            .pointer(pointer)
            .ok_or_else(|| anyhow::anyhow!("docs pointer not found: {}", pointer))?;
        value
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("docs pointer {} is not an array", pointer))?
    } else if let Some(arr) = infer_docs_array(root) {
        arr
    } else {
        anyhow::bail!(
            "could not infer retrieved docs array; pass --docs-pointer (e.g. /retrieval/docs)"
        )
    };

    if docs_array.is_empty() {
        anyhow::bail!("retrieved docs array is empty");
    }

    let mut docs = Vec::with_capacity(docs_array.len());
    for (idx, item) in docs_array.iter().enumerate() {
        let obj = item.as_object().ok_or_else(|| {
            anyhow::anyhow!(
                "retrieved_docs[{}] must be an object with id/text fields",
                idx
            )
        })?;

        let id = first_string_key(
            obj,
            &["id", "doc_id", "document_id", "chunk_id", "source_id"],
        )
        .unwrap_or_else(|| format!("doc_{}", idx + 1));
        let text = first_string_key(obj, &["text", "content", "chunk", "body", "passage"])
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "retrieved_docs[{}] is missing text field (tried text/content/chunk/body/passage)",
                    idx
                )
            })?;
        let score = first_f64_key(obj, &["score", "similarity", "relevance"]);
        let source = first_string_key(obj, &["source", "url", "uri", "path"]);
        let metadata = obj
            .get("metadata")
            .and_then(|v| v.as_object())
            .map(object_to_btree);

        docs.push(RetrievedDoc {
            id,
            text,
            score,
            source,
            metadata,
        });
    }
    Ok(docs)
}

fn infer_docs_array(root: &Value) -> Option<&Vec<Value>> {
    for pointer in DEFAULT_DOCS_POINTERS {
        if let Some(arr) = root.pointer(pointer).and_then(|v| v.as_array()) {
            return Some(arr);
        }
    }
    find_docs_array_recursive(root)
}

fn find_docs_array_recursive(value: &Value) -> Option<&Vec<Value>> {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                if is_docs_key(k) {
                    if let Some(arr) = v.as_array() {
                        if looks_like_docs_array(arr) {
                            return Some(arr);
                        }
                    }
                }
            }
            for v in map.values() {
                if let Some(arr) = find_docs_array_recursive(v) {
                    return Some(arr);
                }
            }
            None
        }
        Value::Array(items) => {
            for item in items {
                if let Some(arr) = find_docs_array_recursive(item) {
                    return Some(arr);
                }
            }
            None
        }
        _ => None,
    }
}

fn is_docs_key(key: &str) -> bool {
    matches!(
        key,
        "retrieved_docs" | "docs" | "documents" | "chunks" | "passages" | "context_docs" | "hits"
    )
}

fn looks_like_docs_array(arr: &[Value]) -> bool {
    if arr.is_empty() {
        return false;
    }
    arr.iter().any(|v| {
        v.as_object().is_some_and(|obj| {
            obj.contains_key("text")
                || obj.contains_key("content")
                || obj.contains_key("chunk")
                || obj.contains_key("body")
                || obj.contains_key("passage")
        })
    })
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Object(map) => {
            for key in ["text", "content", "answer", "output_text", "message"] {
                if let Some(v) = map.get(key) {
                    if let Some(s) = value_to_string(v) {
                        return Some(s);
                    }
                }
            }
            None
        }
        Value::Array(arr) => arr.iter().find_map(value_to_string),
        _ => None,
    }
}

fn first_string_key(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(val) = obj.get(*key) {
            if let Some(s) = value_to_string(val) {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    None
}

fn first_f64_key(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(val) = obj.get(*key) {
            if let Some(n) = val.as_f64() {
                return Some(n);
            }
            if let Some(s) = val.as_str() {
                if let Ok(n) = s.parse::<f64>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

fn object_to_btree(obj: &serde_json::Map<String, Value>) -> BTreeMap<String, Value> {
    obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}
