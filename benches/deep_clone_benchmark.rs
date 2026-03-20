//! Benchmark comparing three deep cloning approaches:
//! 1. Generation-based (new approach) - O(1) via generation check
//! 2. HashMap-based - O(1) hash lookups during clone
//! 3. Index-based two-phase (user's approach) - prepare once, clone with array indexing
//!
//! Test structure: 5 layers, 500 references per layer = 2500 total references

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use graph_clonable_ref::hashmap_based::{HashMapCloneContext, HashMapRefGraph};
use graph_clonable_ref::index_based::{
    clone_refs, prepare_refs, IndexRef, PerformIndexCloningNode, PrepareIndexCloningNode,
};
use graph_clonable_ref::{deep_clone, RefGraph};

const LAYERS: usize = 5;
const REFS_PER_LAYER: usize = 500;

// ============================================================================
// Generation-based setup
// ============================================================================

fn create_generation_based_network() -> Vec<Vec<graph_clonable_ref::GraphRef<i32>>> {
    let mut layers = Vec::with_capacity(LAYERS);

    for layer_idx in 0..LAYERS {
        let graph = RefGraph::new();
        let mut layer = Vec::with_capacity(REFS_PER_LAYER);

        for i in 0..REFS_PER_LAYER {
            let value = (layer_idx * REFS_PER_LAYER + i) as i32;
            let r = graph.create(value);
            layer.push(r);
        }

        // Add clones to simulate shared references
        for i in 0..REFS_PER_LAYER / 2 {
            layer.push(layer[i].clone());
        }

        layers.push(layer);
    }

    layers
}

// ============================================================================
// HashMap-based setup
// ============================================================================

fn create_hashmap_based_network(
) -> Vec<Vec<graph_clonable_ref::hashmap_based::HashMapGraphRef<i32>>> {
    let mut layers = Vec::with_capacity(LAYERS);

    for layer_idx in 0..LAYERS {
        let graph = HashMapRefGraph::new();
        let mut layer = Vec::with_capacity(REFS_PER_LAYER);

        for i in 0..REFS_PER_LAYER {
            let value = (layer_idx * REFS_PER_LAYER + i) as i32;
            let r = graph.create(value);
            layer.push(r);
        }

        layers.push(layer);
    }

    layers
}

// ============================================================================
// Index-based (user's approach) setup
// ============================================================================

fn create_index_based_network() -> (Vec<Vec<IndexRef<i32>>>, Vec<PrepareIndexCloningNode<i32>>) {
    let mut layers = Vec::with_capacity(LAYERS);
    let mut prepare_nodes = Vec::with_capacity(LAYERS);

    for layer_idx in 0..LAYERS {
        let mut layer = Vec::with_capacity(REFS_PER_LAYER + REFS_PER_LAYER / 2);

        // Create unique refs
        for i in 0..REFS_PER_LAYER {
            let value = (layer_idx * REFS_PER_LAYER + i) as i32;
            layer.push(IndexRef::new(value));
        }

        // Add clones to simulate shared references (like generation-based)
        for i in 0..REFS_PER_LAYER / 2 {
            layer.push(layer[i].clone());
        }

        // Prepare node for this layer (done once, reused for each clone)
        let prepare_node = prepare_refs(&layer);
        prepare_nodes.push(prepare_node);

        layers.push(layer);
    }

    (layers, prepare_nodes)
}

// ============================================================================
// Benchmarks
// ============================================================================

/// Main comparison: all three approaches on the 5x500 network.
fn bench_full_network_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_network_5x500");

    // Generation-based
    let gen_network = create_generation_based_network();
    let gen_total: usize = gen_network.iter().map(|l| l.len()).sum();
    group.throughput(Throughput::Elements(gen_total as u64));

    group.bench_function("1_generation_based", |b| {
        b.iter(|| {
            let cloned = deep_clone(black_box(&gen_network));
            black_box(cloned)
        })
    });

    // HashMap-based (lookups during clone)
    let hash_network = create_hashmap_based_network();

    group.bench_function("2_hashmap_based", |b| {
        b.iter(|| {
            let ctx = HashMapCloneContext::new();
            let cloned: Vec<Vec<_>> = hash_network
                .iter()
                .map(|layer| layer.iter().map(|r| ctx.clone_ref(r)).collect())
                .collect();
            black_box(cloned)
        })
    });

    // Index-based two-phase (user's approach)
    let (index_network, prepare_nodes) = create_index_based_network();

    group.bench_function("3_index_two_phase", |b| {
        b.iter(|| {
            let cloned: Vec<Vec<_>> = index_network
                .iter()
                .zip(prepare_nodes.iter())
                .map(|(layer, prepare_node)| clone_refs(layer, prepare_node))
                .collect();
            black_box(cloned)
        })
    });

    // Index-based INCLUDING prepare time (if you had to prepare each time)
    group.bench_function("4_index_with_prepare", |b| {
        b.iter(|| {
            let cloned: Vec<Vec<_>> = index_network
                .iter()
                .map(|layer| {
                    let prepare_node = prepare_refs(layer);
                    clone_refs(layer, &prepare_node)
                })
                .collect();
            black_box(cloned)
        })
    });

    println!("\nNetwork size: {} refs per approach", gen_total);
    group.finish();
}

/// Benchmark just the clone phase (no prepare) for index-based.
fn bench_clone_phase_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("clone_phase_only");

    // Create refs with some duplicates
    let refs: Vec<IndexRef<i32>> = (0..1000)
        .map(|i| IndexRef::new(i))
        .chain((0..500).map(|i| IndexRef::new(i))) // duplicates by value, not by ref
        .collect();

    // Add actual shared refs
    let mut refs_with_sharing: Vec<IndexRef<i32>> = refs.clone();
    for i in 0..500 {
        refs_with_sharing.push(refs[i].clone()); // same Rc
    }

    let prepare_node = prepare_refs(&refs_with_sharing);

    group.throughput(Throughput::Elements(refs_with_sharing.len() as u64));

    group.bench_function("index_clone_only", |b| {
        b.iter(|| {
            let cloned = clone_refs(black_box(&refs_with_sharing), black_box(&prepare_node));
            black_box(cloned)
        })
    });

    // Compare with generation-based
    let graph = RefGraph::new();
    let gen_refs: Vec<_> = (0..1000).map(|i| graph.create(i)).collect();
    let mut gen_refs_with_sharing: Vec<_> = gen_refs.iter().cloned().collect();
    for i in 0..500 {
        gen_refs_with_sharing.push(gen_refs[i].clone());
    }

    group.bench_function("generation_clone", |b| {
        b.iter(|| {
            let cloned = deep_clone(black_box(&gen_refs_with_sharing));
            black_box(cloned)
        })
    });

    group.finish();
}

/// Microbenchmark: per-ref overhead for each approach.
fn bench_per_ref_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_ref_overhead");

    for &size in &[100, 500, 1000, 2500, 5000, 50000] {
        group.throughput(Throughput::Elements(size as u64));

        // Generation-based
        let graph = RefGraph::new();
        let refs: Vec<_> = (0..size / 2).map(|i| graph.create(i as i32)).collect();
        let refs_doubled: Vec<_> = refs.iter().chain(refs.iter()).cloned().collect();

        group.bench_with_input(
            BenchmarkId::new("generation", size),
            &refs_doubled,
            |b, data| {
                b.iter(|| deep_clone(black_box(data)));
            },
        );

        // Index-based (clone only, prepare cached)
        let idx_refs: Vec<_> = (0..size / 2).map(|i| IndexRef::new(i as i32)).collect();
        let idx_doubled: Vec<_> = idx_refs.iter().chain(idx_refs.iter()).cloned().collect();
        let prepare_node = prepare_refs(&idx_doubled);

        group.bench_with_input(
            BenchmarkId::new("index_clone_only", size),
            &(&idx_doubled, &prepare_node),
            |b, (data, prep)| {
                b.iter(|| clone_refs(black_box(data), black_box(prep)));
            },
        );

        // Index-based (including prepare)
        group.bench_with_input(
            BenchmarkId::new("index_with_prepare", size),
            &idx_doubled,
            |b, data| {
                b.iter(|| {
                    let prep = prepare_refs(black_box(data));
                    clone_refs(black_box(data), &prep)
                });
            },
        );

        // HashMap-based
        let hgraph = HashMapRefGraph::new();
        let hrefs: Vec<_> = (0..size / 2).map(|i| hgraph.create(i as i32)).collect();

        group.bench_with_input(BenchmarkId::new("hashmap", size), &hrefs, |b, data| {
            b.iter(|| {
                let ctx = HashMapCloneContext::new();
                let cloned: Vec<_> = data.iter().map(|r| ctx.clone_ref(r)).collect();
                black_box(cloned)
            });
        });
    }

    group.finish();
}

/// Benchmark prepare phase overhead.
fn bench_prepare_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("prepare_phase");

    for &size in &[500, 1000, 2500, 5000] {
        let refs: Vec<_> = (0..size / 2).map(|i| IndexRef::new(i as i32)).collect();
        let refs_doubled: Vec<_> = refs.iter().chain(refs.iter()).cloned().collect();

        group.throughput(Throughput::Elements(refs_doubled.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("prepare", refs_doubled.len()),
            &refs_doubled,
            |b, data| {
                b.iter(|| prepare_refs(black_box(data)));
            },
        );
    }

    group.finish();
}

/// Isolate the pos_array clone overhead in index-based approach.
fn bench_pos_array_clone_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("pos_array_clone");

    for &size in &[500, 1000, 2500, 5000] {
        let refs: Vec<_> = (0..size).map(|i| IndexRef::new(i as i32)).collect();
        let prepare_node = prepare_refs(&refs);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("from_prepare_node", size),
            &prepare_node,
            |b, prep| {
                b.iter(|| {
                    let perform: PerformIndexCloningNode<i32> =
                        PerformIndexCloningNode::from_prepare_node(black_box(prep));
                    black_box(perform)
                });
            },
        );

        // Just the pos_array clone
        group.bench_with_input(
            BenchmarkId::new("pos_array_clone_only", size),
            &prepare_node.pos_array,
            |b, arr| {
                b.iter(|| {
                    let cloned = black_box(arr).clone();
                    black_box(cloned)
                });
            },
        );
    }

    group.finish();
}

/// Real-world scenario: repeated clones with cached prepare node.
fn bench_repeated_clones(c: &mut Criterion) {
    let mut group = c.benchmark_group("repeated_clones_x10");

    // Index-based with cached prepare
    let refs: Vec<_> = (0..500).map(|i| IndexRef::new(i as i32)).collect();
    let refs_doubled: Vec<_> = refs.iter().chain(refs.iter()).cloned().collect();
    let prepare_node = prepare_refs(&refs_doubled);

    group.bench_function("index_10_clones_cached_prepare", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let cloned = clone_refs(black_box(&refs_doubled), black_box(&prepare_node));
                black_box(cloned);
            }
        })
    });

    // Generation-based (no prepare needed)
    let graph = RefGraph::new();
    let gen_refs: Vec<_> = (0..500).map(|i| graph.create(i)).collect();
    let gen_doubled: Vec<_> = gen_refs.iter().chain(gen_refs.iter()).cloned().collect();

    group.bench_function("generation_10_clones", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let cloned = deep_clone(black_box(&gen_doubled));
                black_box(cloned);
            }
        })
    });

    group.finish();
}

/// Benchmark get() and set() throughput at various sizes.
fn bench_get_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_set");

    for &size in &[100, 500, 1000, 5000, 50000] {
        let graph = RefGraph::new();
        let refs: Vec<_> = (0..size).map(|i| graph.create(i as i32)).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("get", size), &refs, |b, data| {
            b.iter(|| {
                for r in data.iter() {
                    black_box(r.get());
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("set", size), &refs, |b, data| {
            b.iter(|| {
                for (i, r) in data.iter().enumerate() {
                    r.set(black_box(i as i32));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_full_network_comparison,
    bench_clone_phase_only,
    bench_per_ref_overhead,
    bench_prepare_overhead,
    bench_pos_array_clone_overhead,
    bench_repeated_clones,
    bench_get_set,
);

criterion_main!(benches);
