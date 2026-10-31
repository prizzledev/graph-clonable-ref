//! A fast cloneable reference that preserves reference structure during deep cloning.
//!
//! Thread-safe (`Send + Sync`) and panic-safe. During [`deep_clone`] each thread records the
//! clone of every graph it meets in a thread-local map, keyed by the original graph's address and
//! emptied when the outermost pass ends (also on unwind). A reference into the same graph as the
//! previous one skips the hash lookup. No reference writes to memory that other threads share, so
//! threads that deep-clone one shared template do not slow each other down.
//!
//! Since 0.3.0 `GraphRef<T>: Clone` requires `T: 'static` (the map stores graphs as `dyn Any`),
//! and [`RefGraph::clear_cache`] does nothing.
//!
//! # Example
//! ```
//! use graph_clonable_ref::{RefGraph, GraphRef, deep_clone};
//!
//! // Create a graph and some references
//! let graph = RefGraph::new();
//! let a = graph.create(42);
//! let b = a.clone(); // b points to same data as a
//!
//! // Modify through one ref, visible through both
//! a.set(100);
//! assert_eq!(b.get(), 100);
//!
//! // Create a struct holding both refs
//! let data = (a, b);
//!
//! // Deep clone preserves structure: a' and b' point to SAME new data
//! let cloned = deep_clone(&data);
//! cloned.0.set(999);
//! assert_eq!(cloned.1.get(), 999); // Both see the change
//! assert_eq!(data.0.get(), 100);   // Original unchanged
//! ```

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

/// A `RefGraph<T>` with its type erased, so graphs of every `T` fit in one map.
type AnyGraph = Arc<dyn Any + Send + Sync>;

/// The clones made by the current deep-clone pass of this thread.
#[derive(Default)]
struct CloneMap {
    /// The most recently used entry: references into one graph skip the hash lookup.
    last: Option<(*const (), AnyGraph)>,
    /// Original address -> (original, kept alive so no other graph gets its address; clone).
    graphs: HashMap<*const (), (AnyGraph, AnyGraph)>,
}

impl CloneMap {
    fn get(&mut self, key: *const ()) -> Option<AnyGraph> {
        match &self.last {
            Some((last, clone)) if *last == key => return Some(Arc::clone(clone)),
            _ => {}
        }
        let clone = Arc::clone(&self.graphs.get(&key)?.1);
        self.last = Some((key, Arc::clone(&clone)));
        Some(clone)
    }

    /// Records `clone` unless the original has a clone already; returns the recorded one.
    fn insert(&mut self, key: *const (), original: AnyGraph, clone: AnyGraph) -> AnyGraph {
        let clone = Arc::clone(&self.graphs.entry(key).or_insert((original, clone)).1);
        self.last = Some((key, Arc::clone(&clone)));
        clone
    }
}

thread_local! {
    static CLONE_DEPTH: Cell<u32> = const { Cell::new(0) };
    /// Counts ended passes, so a clone made across a pass end is not recorded.
    static PASSES_ENDED: Cell<u64> = const { Cell::new(0) };
    static CLONE_MAP: RefCell<CloneMap> = RefCell::default();
}

/// Internal storage for a group of related references.
/// All `GraphRef`s created from the same `RefGraph` share the underlying data.
pub struct RefGraph<T: Send + Sync> {
    data: RwLock<Vec<T>>,
}

impl<T: Send + Sync> RefGraph<T> {
    /// Create a new empty RefGraph.
    pub fn new() -> Arc<Self> {
        Arc::new(RefGraph {
            data: RwLock::new(Vec::new()),
        })
    }

    /// Create a new reference with the given initial value.
    pub fn create(self: &Arc<Self>, value: T) -> GraphRef<T> {
        let mut data = self.data.write();
        let index = data.len();
        data.push(value);
        GraphRef {
            graph: Arc::clone(self),
            index,
        }
    }

    /// Get the number of values stored in this graph.
    pub fn len(&self) -> usize {
        self.data.read().len()
    }

    /// Check if this graph is empty.
    pub fn is_empty(&self) -> bool {
        self.data.read().is_empty()
    }

    /// Does nothing since 0.3.0 and is kept for compatibility: a graph caches no clones any
    /// more; the clone map of a deep clone is thread-local and emptied when the pass ends.
    pub fn clear_cache(&self) {}
}

impl<T: Clone + Send + Sync> RefGraph<T> {
    /// Deep clone this graph, creating independent copies of all data.
    fn deep_clone_graph(&self) -> Arc<RefGraph<T>> {
        let data = self.data.read();
        Arc::new(RefGraph {
            data: RwLock::new(data.clone()),
        })
    }
}

impl<T: Clone + Send + Sync + 'static> RefGraph<T> {
    /// This graph's clone in the current deep-clone pass of this thread, made on first use.
    ///
    /// Touches only thread-local memory and the clone, except on first use, which reads the
    /// data and keeps `self` alive until the pass ends.
    #[inline]
    fn clone_in_pass(self: &Arc<Self>) -> Arc<Self> {
        let key = Arc::as_ptr(self).cast::<()>();
        let clone = match CLONE_MAP.with(|map| map.borrow_mut().get(key)) {
            Some(clone) => clone,
            None => {
                let pass = PASSES_ENDED.with(Cell::get);
                // No borrow held: cloning the data clones the `GraphRef`s inside it.
                let clone = self.deep_clone_graph();
                if PASSES_ENDED.with(Cell::get) != pass {
                    return clone; // the pass ended inside `T::clone`; record nothing
                }
                let clone: AnyGraph = clone;
                CLONE_MAP.with(|map| {
                    map.borrow_mut()
                        .insert(key, Arc::clone(self) as AnyGraph, Arc::clone(&clone))
                })
                // An unused `clone` drops here, outside the borrow, as its values may run code.
            }
        };
        // Invariant: the map keeps the original at `key` alive, so its entry was made by this
        // function for this `T`.
        clone
            .downcast()
            .expect("a clone has the type of its original graph")
    }
}

/// A cloneable reference that preserves structure during deep cloning.
///
/// - Regular `clone()` creates a shallow copy (same underlying data)
/// - When cloned during `deep_clone()`, creates structure-preserving deep copy
pub struct GraphRef<T: Send + Sync> {
    graph: Arc<RefGraph<T>>,
    index: usize,
}

impl<T: Send + Sync> GraphRef<T> {
    /// Get a copy of the referenced value.
    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.graph.data.read()[self.index].clone()
    }

    /// Set the referenced value.
    pub fn set(&self, value: T) {
        self.graph.data.write()[self.index] = value;
    }

    /// Apply a function to the referenced value.
    pub fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut T),
    {
        f(&mut self.graph.data.write()[self.index]);
    }

    /// Get the index of this reference within its graph.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Check if two refs point to the same data (same graph and index).
    pub fn ptr_eq(&self, other: &GraphRef<T>) -> bool {
        Arc::ptr_eq(&self.graph, &other.graph) && self.index == other.index
    }

    /// Check if two refs are in the same graph (may have different indices).
    pub fn same_graph(&self, other: &GraphRef<T>) -> bool {
        Arc::ptr_eq(&self.graph, &other.graph)
    }
}

/// `T: 'static` since 0.3.0: a deep clone stores graphs type-erased in a thread-local map.
impl<T: Clone + Send + Sync + 'static> Clone for GraphRef<T> {
    #[inline]
    fn clone(&self) -> Self {
        let graph = if CLONE_DEPTH.with(Cell::get) > 0 {
            self.graph.clone_in_pass()
        } else {
            // Shallow clone - same graph, same index
            Arc::clone(&self.graph)
        };
        GraphRef {
            graph,
            index: self.index,
        }
    }
}

impl<T: std::fmt::Debug + Clone + Send + Sync> std::fmt::Debug for GraphRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphRef")
            .field("value", &self.get())
            .field("index", &self.index)
            .finish()
    }
}

/// Perform a deep clone that preserves reference structure.
///
/// All `GraphRef`s pointing to the same data before cloning will point
/// to the same (new) data after cloning. Thread-safe and panic-safe.
///
/// # Example
/// ```
/// use graph_clonable_ref::{RefGraph, deep_clone};
///
/// let graph = RefGraph::new();
/// let a = graph.create(1);
/// let b = a.clone();
///
/// let (a2, b2) = deep_clone(&(a.clone(), b.clone()));
///
/// // a2 and b2 point to same NEW data
/// a2.set(42);
/// assert_eq!(b2.get(), 42);
///
/// // Original unchanged
/// assert_eq!(a.get(), 1);
/// ```
pub fn deep_clone<T: Clone>(value: &T) -> T {
    let _guard = begin_deep_clone();
    value.clone()
}

/// Guard that ends the deep-clone pass (depth and clone map) even on panic.
pub struct DeepCloneGuard {
    _private: (),
}

impl Drop for DeepCloneGuard {
    fn drop(&mut self) {
        let depth = CLONE_DEPTH.with(|d| {
            let new = d.get().saturating_sub(1);
            d.set(new);
            new
        });
        if depth == 0 {
            PASSES_ENDED.with(|p| p.set(p.get() + 1));
            // Emptied outside the borrow: dropping a graph runs `Drop` of its values, which may
            // deep-clone. The empty map goes back to keep its capacity for the next pass.
            // try_with: in a thread-local destructor the map may be gone already.
            let Ok(mut map) = CLONE_MAP.try_with(RefCell::take) else {
                return;
            };
            let used = map.graphs.len();
            map.graphs.clear();
            // `clear` costs O(capacity): keep room for the next pass, but not the peak of one
            // exceptionally large pass, which every later pass on this thread would pay for.
            if map.graphs.capacity() > 4 * used.max(16) {
                map.graphs.shrink_to(used);
            }
            map.last = None;
            let _ = CLONE_MAP.try_with(|m| *m.borrow_mut() = map);
        }
    }
}

/// Whether the calling thread is inside a deep-clone pass. Code that relies on shallow clones
/// sharing their data (for example after building an instance from a template) can refuse to
/// run inside one, since every `GraphRef` clone there copies its graph.
pub fn in_deep_clone() -> bool {
    CLONE_DEPTH.with(|depth| depth.get() > 0)
}

/// Begin a deep clone operation manually. Returns a guard that resets state on drop.
///
/// Supports nesting: an inner call joins the pass of the outermost one, which ends when the
/// outermost guard drops. Prefer `deep_clone()` for simple cases.
pub fn begin_deep_clone() -> DeepCloneGuard {
    CLONE_DEPTH.with(|d| d.set(d.get() + 1));
    DeepCloneGuard { _private: () }
}

// ============================================================================
// HashMap-based approach for comparison (slower, single-threaded)
// ============================================================================

pub mod hashmap_based {
    //! HashMap-based deep cloning for comparison.
    //! This is the traditional approach - slower due to hash lookups.

    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    /// Internal storage using HashMap for clone tracking.
    pub struct HashMapRefGraph<T> {
        data: RefCell<Vec<RefCell<T>>>,
    }

    impl<T> HashMapRefGraph<T> {
        pub fn new() -> Rc<Self> {
            Rc::new(HashMapRefGraph {
                data: RefCell::new(Vec::new()),
            })
        }

        pub fn create(self: &Rc<Self>, value: T) -> HashMapGraphRef<T> {
            let mut data = self.data.borrow_mut();
            let index = data.len();
            data.push(RefCell::new(value));
            HashMapGraphRef {
                graph: Rc::clone(self),
                index,
            }
        }
    }

    impl<T: Clone> HashMapRefGraph<T> {
        fn deep_clone_graph(self: &Rc<Self>) -> Rc<HashMapRefGraph<T>> {
            let data = self.data.borrow();
            let cloned_data: Vec<RefCell<T>> = data
                .iter()
                .map(|cell| RefCell::new(cell.borrow().clone()))
                .collect();

            Rc::new(HashMapRefGraph {
                data: RefCell::new(cloned_data),
            })
        }
    }

    pub struct HashMapGraphRef<T> {
        graph: Rc<HashMapRefGraph<T>>,
        index: usize,
    }

    impl<T: Clone> HashMapGraphRef<T> {
        pub fn get(&self) -> T {
            self.graph.data.borrow()[self.index].borrow().clone()
        }

        pub fn set(&self, value: T) {
            *self.graph.data.borrow()[self.index].borrow_mut() = value;
        }
    }

    impl<T> HashMapGraphRef<T> {
        pub fn ptr_eq(&self, other: &HashMapGraphRef<T>) -> bool {
            Rc::ptr_eq(&self.graph, &other.graph) && self.index == other.index
        }
    }

    /// Clone context using HashMap for lookups.
    pub struct HashMapCloneContext<T> {
        map: RefCell<HashMap<*const HashMapRefGraph<T>, Rc<HashMapRefGraph<T>>>>,
    }

    impl<T: Clone> HashMapCloneContext<T> {
        pub fn new() -> Self {
            HashMapCloneContext {
                map: RefCell::new(HashMap::new()),
            }
        }

        /// Clone a single ref using HashMap lookup.
        pub fn clone_ref(&self, r: &HashMapGraphRef<T>) -> HashMapGraphRef<T> {
            let ptr = Rc::as_ptr(&r.graph);
            let mut map = self.map.borrow_mut();

            let new_graph = map
                .entry(ptr)
                .or_insert_with(|| r.graph.deep_clone_graph())
                .clone();

            HashMapGraphRef {
                graph: new_graph,
                index: r.index,
            }
        }
    }

    impl<T: Clone> Default for HashMapCloneContext<T> {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ============================================================================
// Two-phase index-based approach (like user's implementation)
// ============================================================================

pub mod index_based {
    //! Two-phase index-based deep cloning (matching user's approach).
    //!
    //! Phase 1 (Prepare): Walk through all refs, build HashMap lookup and pos_array
    //! Phase 2 (Perform): Walk through again, use pos_array for O(1) lookups
    //!
    //! This avoids HashMap lookups during the actual clone, but has overhead:
    //! - Two passes over all data
    //! - Cloning pos_array for each clone operation
    //! - Pre-allocating ref_array

    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    /// A reference that supports two-phase cloning.
    pub struct IndexRef<T> {
        inner: Rc<RefCell<T>>,
    }

    impl<T> IndexRef<T> {
        pub fn new(value: T) -> Self {
            IndexRef {
                inner: Rc::new(RefCell::new(value)),
            }
        }

        pub fn ptr(&self) -> *const RefCell<T> {
            Rc::as_ptr(&self.inner)
        }
    }

    impl<T: Clone> IndexRef<T> {
        pub fn get(&self) -> T {
            self.inner.borrow().clone()
        }

        pub fn set(&self, value: T) {
            *self.inner.borrow_mut() = value;
        }
    }

    impl<T> Clone for IndexRef<T> {
        fn clone(&self) -> Self {
            IndexRef {
                inner: Rc::clone(&self.inner),
            }
        }
    }

    impl<T> IndexRef<T> {
        pub fn ptr_eq(&self, other: &IndexRef<T>) -> bool {
            Rc::ptr_eq(&self.inner, &other.inner)
        }
    }

    /// Preparation node - built once, stores the reference structure.
    /// Uses HashMap to detect shared references.
    pub struct PrepareIndexCloningNode<T> {
        /// (is_existing, position) for each reference in traversal order
        pub pos_array: Vec<(bool, usize)>,
        /// Maps raw pointer -> position in ref_array
        lookup_map: HashMap<*const RefCell<T>, usize>,
        /// Counter for unique references
        ref_pos_counter: usize,
    }

    impl<T> PrepareIndexCloningNode<T> {
        pub fn new() -> Self {
            Self {
                pos_array: Vec::new(),
                lookup_map: HashMap::new(),
                ref_pos_counter: 0,
            }
        }

        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                pos_array: Vec::with_capacity(capacity),
                lookup_map: HashMap::with_capacity(capacity),
                ref_pos_counter: 0,
            }
        }

        /// Handle a reference during preparation phase.
        /// Records whether it's new or existing, and its position.
        pub fn handle_reference(&mut self, r: &IndexRef<T>) {
            let ptr = r.ptr();
            if let Some(&existing_pos) = self.lookup_map.get(&ptr) {
                // Already seen this reference
                self.pos_array.push((true, existing_pos));
            } else {
                // New reference
                let pos = self.ref_pos_counter;
                self.ref_pos_counter += 1;
                self.lookup_map.insert(ptr, pos);
                self.pos_array.push((false, pos));
            }
        }

        /// Get the number of unique references.
        pub fn unique_count(&self) -> usize {
            self.ref_pos_counter
        }
    }

    impl<T> Default for PrepareIndexCloningNode<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Perform node - created from PrepareNode for each clone operation.
    pub struct PerformIndexCloningNode<T> {
        /// Current position in pos_array
        counter: usize,
        /// New references by position; `None` until the first reference to it is cloned.
        ref_array: Vec<Option<Rc<RefCell<T>>>>,
        /// Cloned from PrepareNode
        pos_array: Vec<(bool, usize)>,
    }

    impl<T: Clone> PerformIndexCloningNode<T> {
        /// Create from a prepare node. This clones the pos_array.
        pub fn from_prepare_node(node: &PrepareIndexCloningNode<T>) -> Self {
            // `None` instead of placeholder values: zeroed memory is not a valid `T` in general
            // (`String`, references, `NonZero*`), and creating one is undefined behaviour.
            let ref_array = vec![None; node.ref_pos_counter];

            Self {
                counter: 0,
                ref_array,
                pos_array: node.pos_array.clone(), // This clone is expensive!
            }
        }

        /// Handle a reference during clone phase.
        /// Must be called in the same order as during preparation.
        pub fn handle_reference(&mut self, orig: &IndexRef<T>) -> IndexRef<T> {
            let pos = self.counter;
            self.counter += 1;

            let (exists, array_pos) = self.pos_array[pos];

            if !exists {
                // First time seeing this reference in this clone
                let new_ref = Rc::new(RefCell::new(orig.inner.borrow().clone()));
                self.ref_array[array_pos] = Some(Rc::clone(&new_ref));
                IndexRef { inner: new_ref }
            } else {
                // Already cloned, return the cached one
                let existing = self.ref_array[array_pos]
                    .as_ref()
                    .expect("a repeated reference is preceded by its first occurrence");
                IndexRef {
                    inner: Rc::clone(existing),
                }
            }
        }
    }

    /// Convenience function to prepare a slice of refs.
    pub fn prepare_refs<T>(refs: &[IndexRef<T>]) -> PrepareIndexCloningNode<T> {
        let mut node = PrepareIndexCloningNode::with_capacity(refs.len());
        for r in refs {
            node.handle_reference(r);
        }
        node
    }

    /// Convenience function to clone refs using a prepare node.
    pub fn clone_refs<T: Clone>(
        refs: &[IndexRef<T>],
        prepare_node: &PrepareIndexCloningNode<T>,
    ) -> Vec<IndexRef<T>> {
        let mut perform_node = PerformIndexCloningNode::from_prepare_node(prepare_node);
        refs.iter()
            .map(|r| perform_node.handle_reference(r))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shallow_clone_same_data() {
        let graph = RefGraph::new();
        let a = graph.create(42);
        let b = a.clone();

        assert!(a.ptr_eq(&b));
        assert_eq!(a.get(), 42);
        assert_eq!(b.get(), 42);

        a.set(100);
        assert_eq!(b.get(), 100);
    }

    #[test]
    fn test_deep_clone_preserves_structure() {
        let graph = RefGraph::new();
        let a = graph.create(42);
        let b = a.clone();

        let (a2, b2) = deep_clone(&(a.clone(), b.clone()));

        // a2 and b2 should point to same new data
        assert!(a2.ptr_eq(&b2));

        // But different from original
        assert!(!a.ptr_eq(&a2));
        assert!(!b.ptr_eq(&b2));

        // Verify independence
        a2.set(999);
        assert_eq!(b2.get(), 999); // Same data
        assert_eq!(a.get(), 42); // Original unchanged
    }

    #[test]
    fn test_deep_clone_multiple_graphs() {
        let graph1 = RefGraph::new();
        let graph2 = RefGraph::new();

        let a1 = graph1.create(1);
        let b1 = a1.clone();
        let a2 = graph2.create(2);
        let b2 = a2.clone();

        let cloned = deep_clone(&(a1.clone(), b1.clone(), a2.clone(), b2.clone()));

        // Same structure preserved within each graph
        assert!(cloned.0.ptr_eq(&cloned.1));
        assert!(cloned.2.ptr_eq(&cloned.3));

        // Different graphs remain different
        assert!(!cloned.0.ptr_eq(&cloned.2));
    }

    #[test]
    fn test_deep_clone_different_indices() {
        let graph = RefGraph::new();
        let a = graph.create(1);
        let b = graph.create(2);
        let c = a.clone(); // Same index as a

        let (a2, b2, c2) = deep_clone(&(a.clone(), b.clone(), c.clone()));

        // a2 and c2 same, b2 different index
        assert!(a2.ptr_eq(&c2));
        assert!(!a2.ptr_eq(&b2));
        assert!(a2.same_graph(&b2));

        // Values preserved
        assert_eq!(a2.get(), 1);
        assert_eq!(b2.get(), 2);
    }

    #[test]
    fn test_nested_struct() {
        #[derive(Clone)]
        struct Inner {
            x: GraphRef<i32>,
            y: GraphRef<i32>,
        }

        #[derive(Clone)]
        struct Outer {
            inner: Inner,
            z: GraphRef<i32>,
        }

        let graph = RefGraph::new();
        let r = graph.create(42);

        let original = Outer {
            inner: Inner {
                x: r.clone(),
                y: r.clone(),
            },
            z: r.clone(),
        };

        let cloned = deep_clone(&original);

        // All three refs in cloned should point to same new data
        assert!(cloned.inner.x.ptr_eq(&cloned.inner.y));
        assert!(cloned.inner.x.ptr_eq(&cloned.z));

        // Independent from original
        cloned.inner.x.set(999);
        assert_eq!(cloned.z.get(), 999);
        assert_eq!(original.z.get(), 42);
    }

    // ========================================================================
    // Thread safety tests
    // ========================================================================

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RefGraph<i32>>();
        assert_send_sync::<GraphRef<i32>>();
        assert_send_sync::<RefGraph<String>>();
        assert_send_sync::<GraphRef<String>>();
    }

    #[test]
    fn test_parallel_deep_clones_of_shared_template() {
        use std::thread;

        // Two graphs; `a` is referenced twice, `b` lives in another graph.
        let first = RefGraph::new();
        let second = RefGraph::new();
        let a = first.create(1usize);
        let template = Arc::new((a.clone(), a, second.create(2usize)));

        let handles: Vec<_> = (0..16)
            .map(|t| {
                let template = Arc::clone(&template);
                thread::spawn(move || {
                    let mut previous: Option<(GraphRef<usize>, GraphRef<usize>, GraphRef<usize>)> =
                        None;
                    for i in 0..500 {
                        let c = deep_clone(template.as_ref());
                        assert!(c.0.ptr_eq(&c.1), "shared reference split");
                        assert!(!c.0.same_graph(&c.2), "graphs merged");
                        assert!(!c.0.same_graph(&template.0), "clone shares the original");
                        if let Some(p) = &previous {
                            assert!(!c.0.same_graph(&p.0), "two deep clones share a graph");
                        }
                        c.0.set(t * 10_000 + i);
                        assert_eq!(c.1.get(), t * 10_000 + i);
                        assert_eq!(c.2.get(), 2);
                        previous = Some(c);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(template.0.get(), 1);
        assert_eq!(template.2.get(), 2);
    }

    #[test]
    fn test_concurrent_deep_clone() {
        use std::thread;

        let graph = RefGraph::new();
        let a = graph.create(42);
        let b = a.clone();
        let c = graph.create(100);

        let data = Arc::new((a.clone(), b.clone(), c.clone()));

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let data = Arc::clone(&data);
                thread::spawn(move || {
                    let cloned = deep_clone(data.as_ref());

                    // Structure preserved: a' and b' point to same new data
                    assert!(cloned.0.ptr_eq(&cloned.1));
                    // c' is in same graph but different index
                    assert!(cloned.0.same_graph(&cloned.2));
                    assert!(!cloned.0.ptr_eq(&cloned.2));

                    // Independence: mutation doesn't affect original
                    cloned.0.set(i * 1000);
                    assert_eq!(cloned.1.get(), i * 1000);
                    assert_eq!(cloned.2.get(), 100);

                    // Original unchanged
                    assert_eq!(data.0.get(), 42);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_read_write() {
        use std::sync::Barrier;
        use std::thread;

        let graph = RefGraph::new();
        let r = graph.create(0i64);
        let barrier = Arc::new(Barrier::new(5));

        // 1 writer thread
        let writer_ref = r.clone();
        let writer_barrier = Arc::clone(&barrier);
        let writer = thread::spawn(move || {
            writer_barrier.wait();
            for i in 0..1000 {
                writer_ref.set(i);
            }
        });

        // 4 reader threads
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let reader_ref = r.clone();
                let reader_barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    reader_barrier.wait();
                    let mut last = -1i64;
                    for _ in 0..1000 {
                        let val = reader_ref.get();
                        // Values should be valid i64 (no torn reads)
                        assert!((0..1000).contains(&val));
                        let _ = last;
                        last = val;
                    }
                })
            })
            .collect();

        writer.join().unwrap();
        for r in readers {
            r.join().unwrap();
        }
    }

    #[test]
    fn test_deep_clone_stress() {
        use std::sync::Barrier;
        use std::thread;

        let graph = RefGraph::new();
        let refs: Vec<_> = (0..10).map(|i| graph.create(i)).collect();
        let data = Arc::new(refs);
        let barrier = Arc::new(Barrier::new(50));

        let handles: Vec<_> = (0..50)
            .map(|_| {
                let data = Arc::clone(&data);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..100 {
                        let cloned = deep_clone(data.as_ref());
                        // All cloned refs should be in the same graph
                        for j in 1..cloned.len() {
                            assert!(cloned[0].same_graph(&cloned[j]));
                        }
                        // Values preserved
                        for (j, r) in cloned.iter().enumerate() {
                            assert_eq!(r.get(), j as i32);
                        }
                        // Independence
                        cloned[0].set(999);
                        assert_eq!(data[0].get(), 0);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    // ========================================================================
    // Panic safety tests
    // ========================================================================

    #[test]
    fn test_panic_resets_generation() {
        use std::panic;

        struct PanicOnClone;

        impl Clone for PanicOnClone {
            fn clone(&self) -> Self {
                panic!("clone panic!");
            }
        }

        // This will panic during deep_clone
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            deep_clone(&PanicOnClone);
        }));
        assert!(result.is_err());

        // The pass should be over — subsequent deep_clone should work
        let graph = RefGraph::new();
        let a = graph.create(42);
        let b = a.clone();
        let (a2, b2) = deep_clone(&(a.clone(), b.clone()));
        assert!(a2.ptr_eq(&b2));
        assert_eq!(a2.get(), 42);
    }

    #[test]
    fn test_panic_nested_recovery() {
        use std::panic;

        // Outer deep_clone that will survive; inner one panics
        let graph = RefGraph::new();
        let a = graph.create(42);

        // Simulate: begin outer deep clone, then panic in inner
        let _result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _outer_guard = begin_deep_clone();
            // Inner deep_clone panics
            let _inner_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let _inner_guard = begin_deep_clone();
                panic!("inner panic");
            }));
            // After inner panic, outer should still work at depth 0
            // (inner guard was dropped by unwinding, decrementing depth)
        }));
        // outer_guard drops here, ending the pass

        // Should be back to clean state
        let b = a.clone();
        let (a2, b2) = deep_clone(&(a.clone(), b.clone()));
        assert!(a2.ptr_eq(&b2));
        assert_eq!(a2.get(), 42);
    }

    // ========================================================================
    // Nesting tests
    // ========================================================================

    #[test]
    fn test_nested_deep_clone() {
        // A type whose Clone impl calls deep_clone internally
        #[derive(Debug)]
        struct Container {
            inner: Vec<GraphRef<i32>>,
        }

        impl Clone for Container {
            fn clone(&self) -> Self {
                // This should work with nesting — inner deep_clone
                // joins the outer pass
                Container {
                    inner: self.inner.clone(),
                }
            }
        }

        let graph = RefGraph::new();
        let a = graph.create(10);
        let b = a.clone();

        let container = Container {
            inner: vec![a.clone(), b.clone()],
        };

        let cloned = deep_clone(&container);
        assert!(cloned.inner[0].ptr_eq(&cloned.inner[1]));
        cloned.inner[0].set(999);
        assert_eq!(cloned.inner[1].get(), 999);
        assert_eq!(container.inner[0].get(), 10);
    }

    #[test]
    fn test_begin_deep_clone_nested_guards() {
        let graph = RefGraph::new();
        let a = graph.create(1);
        let b = a.clone();

        {
            let _outer = begin_deep_clone();
            let cloned_a = a.clone(); // deep clone
            {
                let _inner = begin_deep_clone();
                let cloned_b = b.clone(); // same pass as outer
                // Both should point to same new graph
                assert!(cloned_a.same_graph(&cloned_b));
            }
            // Inner guard dropped, but depth > 0, so the pass goes on
            let cloned_b2 = b.clone();
            assert!(cloned_a.ptr_eq(&cloned_b2));
        }
        // Outer guard dropped, depth == 0, pass ended

        // Now a regular clone should be shallow
        let shallow = a.clone();
        assert!(a.ptr_eq(&shallow));
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn test_multiple_graph_types() {
        #[derive(Clone)]
        struct Mixed {
            nums: Vec<GraphRef<i32>>,
            strs: Vec<GraphRef<String>>,
        }

        let g1 = RefGraph::new();
        let g2 = RefGraph::new();

        let n = g1.create(42);
        let s = g2.create("hello".to_string());

        let data = Mixed {
            nums: vec![n.clone(), n.clone()],
            strs: vec![s.clone(), s.clone()],
        };

        let cloned = deep_clone(&data);
        assert!(cloned.nums[0].ptr_eq(&cloned.nums[1]));
        assert!(cloned.strs[0].ptr_eq(&cloned.strs[1]));

        cloned.nums[0].set(999);
        assert_eq!(cloned.nums[1].get(), 999);
        assert_eq!(data.nums[0].get(), 42);

        cloned.strs[0].set("world".to_string());
        assert_eq!(cloned.strs[1].get(), "world");
        assert_eq!(data.strs[0].get(), "hello");
    }

    #[test]
    fn test_large_graph() {
        let graph = RefGraph::new();
        let refs: Vec<_> = (0..10_000).map(|i| graph.create(i)).collect();

        // Add some shared refs
        let mut all_refs = refs.clone();
        all_refs.extend(refs[..5_000].iter().cloned());

        let cloned = deep_clone(&all_refs);

        assert_eq!(cloned.len(), 15_000);

        // Check sharing is preserved
        for i in 0..5_000 {
            assert!(cloned[i].ptr_eq(&cloned[10_000 + i]));
        }

        // Check values
        for (i, r) in cloned[..10_000].iter().enumerate() {
            assert_eq!(r.get(), i as i32);
        }

        // Independence
        cloned[0].set(-1);
        assert_eq!(all_refs[0].get(), 0);
    }

    // ========================================================================
    // Thread-local clone map tests
    // ========================================================================

    /// Whether this thread's clone map holds nothing: no clone and no original kept alive.
    fn clone_map_is_empty() -> bool {
        CLONE_MAP.with(|m| {
            let m = m.borrow();
            m.graphs.is_empty() && m.last.is_none()
        })
    }

    #[test]
    fn test_thread_reuse_leaves_no_stale_state() {
        // One thread runs pass after pass, as a pool thread does; each graph is dropped before
        // the next one is made, so the next one may get its address.
        std::thread::spawn(|| {
            let mut previous: Option<GraphRef<i32>> = None;
            for round in 0..3 {
                let graph = RefGraph::new();
                let a = graph.create(round);
                let count = Arc::strong_count(&graph);
                let (a2, b2) = deep_clone(&(a.clone(), a.clone()));
                assert!(clone_map_is_empty());
                assert_eq!(Arc::strong_count(&graph), count, "pass kept the original");
                assert_eq!(Arc::strong_count(&a2.graph), 2, "pass kept the clone");
                assert!(a2.ptr_eq(&b2));
                assert_eq!(a2.get(), round, "clone of an earlier pass");
                if let Some(p) = &previous {
                    assert!(!a2.same_graph(p));
                }
                previous = Some(a2);
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn test_original_dropped_mid_pass_keeps_its_address() {
        let _pass = begin_deep_clone();
        let a = RefGraph::new().create(1);
        let a2 = a.clone();
        drop(a); // only the clone map holds the original graph now
        // Graphs of the same size: the allocator would hand out the dropped one's address.
        for i in 2..100 {
            let b2 = RefGraph::new().create(i).clone();
            assert!(
                !b2.same_graph(&a2),
                "new graph got the clone of a dropped one"
            );
            assert_eq!(b2.get(), i);
        }
    }

    #[test]
    fn test_panic_mid_pass_then_new_pass() {
        use std::panic;

        #[derive(Debug)]
        struct PanicOnClone;
        impl Clone for PanicOnClone {
            fn clone(&self) -> Self {
                panic!("clone panic!");
            }
        }

        let graph = RefGraph::new();
        let a = graph.create(7);
        let p = RefGraph::new().create(PanicOnClone);
        let count = Arc::strong_count(&graph);
        // `a`'s graph is recorded, then cloning `p`'s graph panics.
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            deep_clone(&(a.clone(), p.clone()));
        }));
        assert!(result.is_err());
        assert!(clone_map_is_empty());
        assert_eq!(Arc::strong_count(&graph), count);

        let (a2, b2) = deep_clone(&(a.clone(), a.clone()));
        assert!(a2.ptr_eq(&b2));
        assert!(!a2.same_graph(&a));
        assert_eq!(a2.get(), 7);
        assert_eq!(Arc::strong_count(&a2.graph), 2);
    }

    #[test]
    fn test_many_graphs_in_one_pass() {
        // Each reference is in another graph than the one before (types alternate, too), so
        // every reference misses the last-used entry and takes the hash lookup.
        let n = 1000;
        let nums: Vec<_> = (0..n).map(|i| RefGraph::new().create(i)).collect();
        let strs: Vec<_> = (0..n)
            .map(|i| RefGraph::new().create(i.to_string()))
            .collect();
        let pairs = || nums.iter().cloned().zip(strs.iter().cloned());
        let template: Vec<_> = pairs().chain(pairs().rev()).collect();

        let cloned = deep_clone(&template);

        assert!(clone_map_is_empty());
        for i in 0..n {
            let (num, s) = &cloned[i];
            let (num_again, s_again) = &cloned[2 * n - 1 - i];
            assert!(
                num.ptr_eq(num_again) && s.ptr_eq(s_again),
                "shared reference split"
            );
            assert!(!num.same_graph(&nums[i]) && !s.same_graph(&strs[i]));
            assert_eq!((num.get(), s.get()), (i, i.to_string()));
            assert_eq!(Arc::strong_count(&num.graph), 2, "pass kept the clone");
            assert_eq!(
                Arc::strong_count(&nums[i].graph),
                3,
                "pass kept the original"
            );
            if i > 0 {
                assert!(!num.same_graph(&cloned[i - 1].0), "graphs merged");
            }
        }
    }

    #[test]
    fn test_graph_of_graph_refs() {
        // Cloning the outer graph's values clones references into the inner graph, so the
        // clone map is used again while the outer graph is being cloned.
        let inner = RefGraph::new().create(1);
        let outer = RefGraph::new().create(inner.clone());

        let (outer2, inner2) = deep_clone(&(outer.clone(), inner.clone()));

        assert!(outer2.get().ptr_eq(&inner2));
        assert!(!inner2.ptr_eq(&inner));
        inner2.set(5);
        assert_eq!(inner.get(), 1);
    }

    #[test]
    fn in_deep_clone_reports_the_pass() {
        assert!(!in_deep_clone());
        let guard = begin_deep_clone();
        assert!(in_deep_clone());
        drop(guard);
        assert!(!in_deep_clone());
    }
}
