use crate::model::Document;

pub fn normalize_documents(docs: Vec<Document>) -> Vec<Document> {
    docs.into_iter()
        .map(|mut d| {
            d.text = normalize(&d.text);
            d
        })
        .collect()
}

fn normalize(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut last_blank = false;
    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            if !last_blank {
                cleaned.push('\n');
                last_blank = true;
            }
            continue;
        }
        cleaned.push_str(trimmed.trim_matches('\u{feff}'));
        cleaned.push('\n');
        last_blank = false;
    }
    cleaned.trim().to_string()
}
