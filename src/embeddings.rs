use crate::model::{ChunkEmbedding, Corpus};
use anyhow::Result;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub trait Embedder: Sync + Send {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[derive(Clone)]
pub struct NullEmbedder {
    dim: usize,
}

impl Default for NullEmbedder {
    fn default() -> Self {
        Self { dim: 64 }
    }
}

impl NullEmbedder {
    fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0f32; self.dim];
        for token in text.split_whitespace() {
            let idx = self.hash_token(token) % self.dim as u64;
            vec[idx as usize] += 1.0;
        }
        vec
    }

    fn hash_token(&self, token: &str) -> u64 {
        let mut h = DefaultHasher::new();
        token.hash(&mut h);
        h.finish()
    }
}

impl Embedder for NullEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_text(t)).collect())
    }
}

pub fn embed_corpus(corpus: &Corpus, embedder: &dyn Embedder) -> Result<Vec<ChunkEmbedding>> {
    let texts: Vec<String> = corpus.chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = embedder.embed_batch(&texts)?;

    Ok(corpus
        .chunks
        .iter()
        .zip(embeddings.into_iter())
        .map(|(chunk, vector)| ChunkEmbedding {
            chunk_id: chunk.chunk_id.clone(),
            vector,
        })
        .collect())
}
