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

This crate uses **generation-based tracking** instead of HashMap lookups:

- Each `RefGraph` caches its clone with a globally unique generation number
- During `deep_clone()`, a unique generation is assigned via atomic counter
- References check if their graph's cached clone matches the current generation
- First reference triggers the clone, subsequent references get O(1) cache hits
- Cache entries use `Weak` refs and auto-expire when no longer referenced

**Result: ~7 nanoseconds per reference** — thread-safe, panic-safe, and 2.5x faster than HashMap.

## Features

- **Thread-safe** — `Send + Sync`, works with `Arc`, actix-web, tokio, rayon, etc.
- **Panic-safe** — `DeepCloneGuard` with depth tracking resets state correctly on unwind
- **Fast** — generation-based O(1) cache hits, ~7ns per reference
- **Nestable** — `deep_clone` inside `deep_clone` works correctly
- **Zero boilerplate** — works with `#[derive(Clone)]`, no special traits needed
- **No `'static` bound** — works with any lifetime

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
graph-clonable-ref = "0.1"
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
graph.clear_cache();                      // free clone cache memory
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
let b2 = b.clone();  // deep clone, same generation
// guard dropped -> back to shallow clone behavior
```

## Performance

Benchmarks on a network with 5 layers x 500 references (3750 total with sharing):

| Approach | Time | Per Reference |
|----------|------|---------------|
| **Generation-based (this crate)** | ~26 µs | ~7 ns |
| HashMap-based | ~64 µs | ~17 ns |
| Two-phase index (clone only) | ~105 µs | ~28 ns |
| Two-phase index (with prepare) | ~191 µs | ~51 ns |

Run benchmarks yourself:

```bash
cargo bench
```

## How It Works

1. **Normal clone (`a.clone()`)**: Returns a new `GraphRef` pointing to the same data (shallow, like `Arc::clone`)

2. **Deep clone (`deep_clone(&x)`)**:
   - Assigns a globally unique generation via `AtomicU64`
   - Calls `x.clone()` which triggers special behavior
   - Each `GraphRef::clone()` checks: does my graph have a cached clone for this generation?
     - **Yes** (read lock): Return a ref to the cached clone — O(1)
     - **No** (write lock): Clone the graph, cache with `Weak` ref, return new ref
   - `DeepCloneGuard` resets state on drop (even on panic)

Cache entries use `Weak<RefGraph<T>>` so they auto-expire when the cloned graph is dropped. Dead entries are cleaned up lazily on cache-miss writes.

## License

MIT
