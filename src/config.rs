#[derive(Debug, Clone)]
pub struct Config {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub top_k: usize,
    pub dominant_threshold: f32,
    pub required_metadata: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_size: 400,
            chunk_overlap: 40,
            max_tokens: 1200,
            min_tokens: 60,
            top_k: 5,
            dominant_threshold: 0.2,
            required_metadata: vec!["product".into(), "region".into(), "version".into()],
        }
    }
}
