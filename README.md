# graph-clonable-ref

[![Crates.io](https://img.shields.io/crates/v/graph-clonable-ref.svg)](https://crates.io/crates/graph-clonable-ref)
[![Documentation](https://docs.rs/graph-clonable-ref/badge.svg)](https://docs.rs/graph-clonable-ref)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/prizzledev/graph-clonable-ref/actions/workflows/rust.yml/badge.svg)](https://github.com/prizzledev/graph-clonable-ref/actions)

A fast, thread-safe cloneable reference type for Rust that preserves reference structure during deep cloning.

## The Problem

When you have shared references (like `Arc<RwLock<T>>`) in a struct and clone it, each reference is cloned independently. References that pointed to the same data before cloning now point to different copies — structure is lost.

Traditional solutions use a `HashMap<*const T, Arc<T>>` during cloning to track which references have been cloned. This works but is slow (~17ns per reference lookup).

## The Solution

This crate gives each deep clone a **thread-local clone map** keyed by graph, not by reference:

- During `deep_clone()`, the thread records the clone of every graph it meets, keyed by the original graph's address
- A reference into the same graph as the reference before it takes the clone from a last-used slot, without a hash lookup; a reference into another graph takes one hash lookup
- Only the first reference into a graph touches the original graph: one read lock to clone its data and one reference count increment to keep it alive for the pass, so its address stays unique
- Every further reference into that graph touches only thread-local memory and the new graph, so threads that deep-clone one shared template scale with the number of cores
- The map is emptied when the outermost `deep_clone()` ends, also when a clone panics; it keeps its capacity for the next pass unless one pass was much larger than the current one

Up to 0.2, each graph cached its clones behind a lock that every cloned reference took. Threads that deep-cloned one shared template slowed each other down: with one graph of 2048 values, 16 threads together managed 7% of the deep clones per second of one thread (`benches/parallel_scaling.rs`).

## Features

- **Thread-safe** — `Send + Sync`, works with `Arc`, actix-web, tokio, rayon, etc.
- **Scales across threads** — after the first reference into a graph, a cloned reference touches only thread-local memory and its new graph
- **Panic-safe** — `DeepCloneGuard` with depth tracking ends the pass and empties the clone map on unwind
- **Fast** — no hash lookup for consecutive references into one graph, one lookup otherwise
- **Nestable** — `deep_clone` inside `deep_clone` works correctly
- **Zero boilerplate** — works with `#[derive(Clone)]`, no special traits needed
- **`T: 'static`** — `GraphRef<T>` is `Clone` only for `T: 'static`, since the clone map stores graphs type-erased (`dyn Any`)

### Upgrading from 0.2

- `GraphRef<T>: Clone` now requires `T: 'static`, for shallow clones too (both go through `Clone`); a `GraphRef<&'a str>` with a borrowed, non-`'static` lifetime can no longer be cloned, and generic code that clones a `GraphRef<T>` needs a `T: 'static` bound
- `RefGraph::clear_cache()` does nothing; it is kept so existing calls still compile
- Deep-cloning graph references inside a thread-local destructor can panic (and so abort) once the crate's own thread-local map has been destroyed; deep-cloning plain values there works
- Single-threaded, templates made of many tiny graphs (one value each) deep-clone about 30% slower than in 0.2, because each graph costs a hash map insert; one graph with many values is faster

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
graph-clonable-ref = "0.3"
```

## Usage

### Basic Example

```rust
use graph_clonable_ref::{RefGraph, GraphRef, deep_clone};

// Create a graph (container for related references)
let graph = RefGraph::new();

// Create references within the graph
let a = graph.create(42);
let b = a.clone();  // b points to same data as a

// Verify they share data
a.set(100);
assert_eq!(b.get(), 100);

// Deep clone preserves structure
let (a2, b2) = deep_clone(&(a.clone(), b.clone()));

// a2 and b2 point to the SAME new data
a2.set(999);
assert_eq!(b2.get(), 999);

// Original is unaffected
assert_eq!(a.get(), 100);
```

### In Structs

```rust
use graph_clonable_ref::{RefGraph, GraphRef, deep_clone};

#[derive(Clone)]
struct NeuralNetwork {
    weights: Vec<GraphRef<f64>>,
    biases: Vec<GraphRef<f64>>,
    tied_weights: Vec<GraphRef<f64>>,  // shares data with weights
}

impl NeuralNetwork {
    fn new() -> Self {
        let graph = RefGraph::new();
        let weights: Vec<_> = (0..100).map(|_| graph.create(0.0)).collect();
        let biases: Vec<_> = (0..10).map(|_| graph.create(0.0)).collect();
        let tied_weights = weights[0..10].to_vec();
        NeuralNetwork { weights, biases, tied_weights }
    }
}

let net = NeuralNetwork::new();
let net2 = deep_clone(&net);

// net2's tied_weights still reference the same data as net2's weights
// but are independent from net's weights
```

### Thread-Safe Usage (actix-web)

```rust
use actix_web::{web, App, HttpServer, HttpResponse};
use graph_clonable_ref::{RefGraph, GraphRef, deep_clone};

#[derive(Clone)]
struct AppState {
    weights: Vec<GraphRef<f64>>,
    tied_weights: Vec<GraphRef<f64>>,
}

async fn train_handler(data: web::Data<AppState>) -> HttpResponse {
    // Each request gets its own independent copy
    let mut local = deep_clone(data.get_ref());

    local.weights[0].set(1.0);
    assert_eq!(local.tied_weights[0].get(), 1.0); // still tied

    HttpResponse::Ok().finish()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let graph = RefGraph::new();
    let weights: Vec<_> = (0..100).map(|i| graph.create(i as f64)).collect();
    let tied_weights = weights[..10].to_vec();

    let state = web::Data::new(AppState { weights, tied_weights });

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/train", web::post().to(train_handler))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

### Multiple Graphs

References from different graphs remain independent:

```rust
use graph_clonable_ref::{RefGraph, GraphRef, deep_clone};

let graph1 = RefGraph::new();
let graph2 = RefGraph::new();

let a = graph1.create(1);
let b = graph1.create(2);
let c = graph2.create(3);

let (a2, b2, c2) = deep_clone(&(a.clone(), b.clone(), c.clone()));

// a2 and b2 share the same cloned graph
assert!(a2.same_graph(&b2));

// c2 is in a different cloned graph
assert!(!a2.same_graph(&c2));
```

## API Reference

### `RefGraph<T: Send + Sync>`

Container for a group of related references.

```rust
let graph = RefGraph::new();              // -> Arc<RefGraph<T>>
let r: GraphRef<i32> = graph.create(42);  // create a reference
let count = graph.len();                  // number of values
graph.clear_cache();                      // no-op since 0.3 (kept for compatibility)
```

### `GraphRef<T: Send + Sync>`

A cloneable reference that preserves structure during deep cloning.

```rust
let value = r.get();          // get value (requires T: Clone)
r.set(100);                   // set value
r.update(|v| *v += 1);       // update with function
a.ptr_eq(&b);                // same graph and index?
a.same_graph(&b);            // same graph?
r.index();                   // index within graph
```

### `deep_clone<T: Clone>(value: &T) -> T`

Performs a structure-preserving deep clone. Thread-safe and panic-safe.

### `begin_deep_clone() -> DeepCloneGuard`

For manual control over the cloning scope (advanced). Supports nesting.

```rust
let _guard = begin_deep_clone();
let a2 = a.clone();  // deep clone
let b2 = b.clone();  // deep clone, same pass
// guard dropped -> back to shallow clone behavior
```

## Performance

Benchmarks on a network with 5 layers x 500 references (3750 total with sharing), measured with 0.2 (generation-based per-graph cache) and not yet repeated for the thread-local map of 0.3:

| Approach | Time | Per Reference |
|----------|------|---------------|
| **Generation-based (0.2)** | ~26 µs | ~7 ns |
| HashMap-based | ~64 µs | ~17 ns |
| Two-phase index (clone only) | ~105 µs | ~28 ns |
| Two-phase index (with prepare) | ~191 µs | ~51 ns |

Run benchmarks yourself:

```bash
cargo bench                              # single thread, compared with other approaches
cargo bench --bench parallel_scaling     # deep clones per second for 1 to 16 threads
```

## How It Works

1. **Normal clone (`a.clone()`)**: Returns a new `GraphRef` pointing to the same data (shallow, like `Arc::clone`)

2. **Deep clone (`deep_clone(&x)`)**:
   - Raises a thread-local depth counter and calls `x.clone()`
   - Each `GraphRef::clone()` sees the depth and asks the thread-local clone map for the clone of its graph:
     - **Same graph as the reference before**: taken from the last-used slot, no hash lookup
     - **Graph met before in this pass**: one hash lookup by the original graph's address
     - **First reference into the graph**: read the data once, clone it into a new graph, and record it together with a strong reference to the original, so no other graph can get that address during the pass
   - `DeepCloneGuard` ends the pass when the outermost guard drops (even on panic): the map is emptied, releasing originals and clones

A `deep_clone` inside another one joins the outer pass. Graphs cache nothing, and the map holds nothing between passes; it keeps its capacity on each thread for the next pass.

## License

MIT
