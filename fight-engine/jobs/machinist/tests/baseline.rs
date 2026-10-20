#[path = "../../../tests/common.rs"]
mod common;
#[path = "../../../tests/common/jobs.rs"]
mod jobs;

use std::path::Path;
use std::sync::Arc;

use fight_engine::config::SchemaConfig;
use fight_engine::contracts::JobRegistry;
use fight_engine_machinist::Machinist;

#[test]
fn machinist_matches_real_baseline() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let fixture =
        std::fs::read_to_string(root.join("fight-engine/tests/fixtures/jobs_machinist.json"))
            .unwrap();
    let mut registry = JobRegistry::default();
    registry.register::<Machinist>().unwrap();
    jobs::run_fixture(
        &serde_json::from_str(&fixture).unwrap(),
        &registry,
        Arc::new(SchemaConfig::load(root).unwrap()),
    );
}
