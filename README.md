# graph-clonable-ref

A fast cloneable reference type for Rust that preserves reference structure during deep cloning.

## The Problem

When you have shared references (like `Rc<RefCell<T>>`) in a struct and clone it, each reference is cloned independently. This means references that pointed to the same data before cloning now point to different copies:

```rust
// Traditional Rc behavior
let a = Rc::new(RefCell::new(42));
let b = Rc::clone(&a);  // a and b point to same data

// After cloning a struct containing both...
// a' and b' point to DIFFERENT data - structure is lost!
```

Traditional solutions use a `HashMap<*const T, Rc<T>>` during cloning to track which references have been cloned. This works but is slow (~25ns per reference lookup).

## The Solution

This crate uses **generation-based tracking** instead of HashMap lookups:

- Each `RefGraph` caches its clone along with a generation number
- During `deep_clone()`, a thread-local generation counter is incremented
- References check if their graph's cached clone matches the current generation
- First reference triggers the clone, subsequent references get O(1) cache hits

**Result: ~6 nanoseconds per reference** (4x faster than HashMap, 8x faster than two-phase index approaches)

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
graph-clonable-ref = { path = "." }  # or publish to crates.io
```

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
assert_eq!(b.get(), 100);  // Both see the change

// Deep clone preserves structure
let (a2, b2) = deep_clone(&(a.clone(), b.clone()));

// a2 and b2 point to the SAME new data
a2.set(999);
assert_eq!(b2.get(), 999);  // Both see the change

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
    // Shared references for weight tying
    tied_weights: Vec<GraphRef<f64>>,
}

impl NeuralNetwork {
    fn new() -> Self {
        let graph = RefGraph::new();

        let weights: Vec<_> = (0..100)
            .map(|_| graph.create(0.0))
            .collect();

        let biases: Vec<_> = (0..10)
            .map(|_| graph.create(0.0))
            .collect();

        // Tie some weights (they share the same underlying data)
        let tied_weights = weights[0..10].iter().cloned().collect();

        NeuralNetwork { weights, biases, tied_weights }
    }

    fn deep_clone(&self) -> Self {
        deep_clone(self)
    }
}

let net = NeuralNetwork::new();
let net2 = net.deep_clone();

// net2's tied_weights still reference the same data as net2's weights
// but are independent from net's weights
```

### Multiple Graphs

References from different graphs remain independent:

```rust
let graph1 = RefGraph::new();
let graph2 = RefGraph::new();

let a = graph1.create(1);
let b = graph1.create(2);
let c = graph2.create(3);

// a and b are in the same graph, c is separate
let (a2, b2, c2) = deep_clone(&(a.clone(), b.clone(), c.clone()));

// a2 and b2 share the same cloned graph
assert!(a2.same_graph(&b2));

// c2 is in a different cloned graph
assert!(!a2.same_graph(&c2));
```

## API Reference

### `RefGraph<T>`

Container for a group of related references.

```rust
// Create a new graph
let graph = RefGraph::new();

// Create a reference with an initial value
let r: GraphRef<i32> = graph.create(42);

// Get number of values in the graph
let count = graph.len();

// Clear the clone cache (optional, for memory management)
graph.clear_cache();
```

### `GraphRef<T>`

A cloneable reference that preserves structure during deep cloning.

```rust
// Get the value (requires T: Clone)
let value = r.get();

// Set the value
r.set(100);

// Update with a function
r.update(|v| *v += 1);

// Check if two refs point to the same data
if a.ptr_eq(&b) { /* same graph and index */ }

// Check if two refs are in the same graph
if a.same_graph(&b) { /* same graph, possibly different indices */ }

// Get the index within the graph
let idx = r.index();
```

### `deep_clone<T: Clone>(value: &T) -> T`

Performs a structure-preserving deep clone.

```rust
let cloned = deep_clone(&original);
```

### `begin_deep_clone() -> DeepCloneGuard`

For manual control over the cloning scope (advanced usage):

```rust
let _guard = begin_deep_clone();
// All GraphRef::clone() calls within this scope will deep clone
let a2 = a.clone();
let b2 = b.clone();
// Guard is dropped, returning to normal shallow clone behavior
```

## Performance

Benchmarks on a network with 5 layers x 500 references (3750 total):

| Approach | Time | Per Reference |
|----------|------|---------------|
| **Generation-based (this crate)** | 24 µs | ~6 ns |
| HashMap-based | 64 µs | ~17 ns |
| Two-phase index (clone only) | 105 µs | ~28 ns |
| Two-phase index (with prepare) | 191 µs | ~51 ns |

Run benchmarks yourself:

```bash
cargo bench
```

## How It Works

1. **Normal clone (`a.clone()`)**: Returns a new `GraphRef` pointing to the same data (shallow clone, like `Rc::clone`)

2. **Deep clone (`deep_clone(&x)`)**:
   - Increments a thread-local generation counter
   - Calls `x.clone()` which triggers special behavior
   - Each `GraphRef::clone()` checks: does my graph have a cached clone for this generation?
     - **Yes**: Return a ref to the cached clone (O(1))
     - **No**: Clone the entire graph, cache it, return ref to new clone
   - Resets the generation counter

The key insight is that the **first reference to each graph pays the cost** of cloning all the graph's data, while **all subsequent references** from the same graph get an O(1) cache hit.

## Thread Safety

This crate uses `Rc` and `RefCell`, so it is **not thread-safe**. For multi-threaded usage, you would need to adapt it to use `Arc` and appropriate synchronization.

## License

MIT or Apache-2.0, at your option.
