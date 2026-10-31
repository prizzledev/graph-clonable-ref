//! Parallel scaling: several threads deep-clone the same shared template, as a compiler that
//! instantiates one template per request does. Prints deep clones per second for 1 to 16
//! threads; the same 4096 references are spread over 1 to 2048 graphs.
//!
//! Run: `cargo bench --bench parallel_scaling` (BENCH_SECS sets the time per cell, default 1).

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use graph_clonable_ref::{GraphRef, RefGraph, deep_clone};

#[derive(Clone)]
struct Template {
    #[allow(dead_code)] // only cloned
    refs: Vec<GraphRef<usize>>,
}

/// `graphs` graphs with `per_graph` values each; every value is referenced twice.
fn template(graphs: usize, per_graph: usize) -> Template {
    let mut refs = Vec::with_capacity(graphs * per_graph * 2);
    for g in 0..graphs {
        let graph = RefGraph::new();
        for i in 0..per_graph {
            let r = graph.create(g * per_graph + i);
            refs.push(r.clone());
            refs.push(r);
        }
    }
    Template { refs }
}

fn clones_per_sec(template: &Arc<Template>, threads: usize, time: Duration) -> f64 {
    let start = Instant::now();
    let workers: Vec<_> = (0..threads)
        .map(|_| {
            let template = Arc::clone(template);
            std::thread::spawn(move || {
                let mut done = 0u64;
                while start.elapsed() < time {
                    black_box(deep_clone(&*template));
                    done += 1;
                }
                done
            })
        })
        .collect();
    let total: u64 = workers.into_iter().map(|w| w.join().unwrap()).sum();
    total as f64 / start.elapsed().as_secs_f64()
}

fn main() {
    let secs = std::env::var("BENCH_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let time = Duration::from_secs_f64(secs);
    let threads = [1, 2, 4, 8, 16];
    println!(
        "| graphs x values | {} | 16 vs 1 |",
        threads.map(|t| format!("{t} thr")).join(" | ")
    );
    println!("|---|{}---:|", "---:|".repeat(threads.len()));
    for (graphs, per_graph) in [(1, 2048), (16, 128), (256, 8), (2048, 1)] {
        let template = Arc::new(template(graphs, per_graph));
        let rates: Vec<f64> = threads
            .iter()
            .map(|&t| clones_per_sec(&template, t, time))
            .collect();
        let cells: Vec<String> = rates.iter().map(|r| format!("{r:.0}")).collect();
        println!(
            "| {graphs} x {per_graph} | {} | {:.1}x |",
            cells.join(" | "),
            rates[rates.len() - 1] / rates[0]
        );
    }
}
