use assert_cmd::Command;
use predicates::str::contains;
use serde_json::json;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn trace_run_reports_evidence_chain() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_trace_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let run = dir.join("run.json");

    fs::write(
        &run,
        serde_json::to_vec_pretty(&json!({
            "question": "Can I get a refund after 90 days?",
            "answer": "Refunds are only allowed within 30 days.",
            "retrieved_docs": [
                {
                    "id": "shipping",
                    "text": "Shipping takes five business days.",
                    "score": 0.60,
                    "source": "shipping.md"
                },
                {
                    "id": "refund_policy",
                    "text": "Refunds are only allowed within 30 days for eligible defects.",
                    "score": 0.91,
                    "source": "refund.md",
                    "metadata": {
                        "chunk_id": "chunk_3",
                        "vector_id": "vec_3"
                    }
                }
            ]
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("raglens").unwrap();
    cmd.arg("trace")
        .arg("--run")
        .arg(&run)
        .assert()
        .success()
        .stdout(contains("RAGLens Trace"))
        .stdout(contains("refund_policy"))
        .stdout(contains("chunk=chunk_3"))
        .stdout(contains("vector=vec_3"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn trace_run_can_emit_json() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_trace_json_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let run = dir.join("run.json");

    fs::write(
        &run,
        serde_json::to_vec_pretty(&json!({
            "question": "Can I get a refund after 90 days?",
            "answer": "Refunds are only allowed within 30 days.",
            "retrieved_docs": [
                {
                    "id": "refund_policy",
                    "text": "Refunds are only allowed within 30 days for eligible defects.",
                    "score": 0.91,
                    "source": "refund.md"
                }
            ]
        }))
        .unwrap(),
    )
    .unwrap();

    let output = Command::cargo_bin("raglens")
        .unwrap()
        .arg("trace")
        .arg("--run")
        .arg(&run)
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let value: serde_json::Value = serde_json::from_slice(&output).unwrap();

    assert_eq!(value["best_evidence"]["doc_id"], "refund_policy");

    let _ = fs::remove_dir_all(dir);
}
