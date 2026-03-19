//! A fast cloneable reference that preserves reference structure during deep cloning.
//!
//! Traditional deep cloning with HashMap lookup is slow (100+ ns per lookup).
//! This crate uses generation-based tracking for O(1) reference resolution during cloning.
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

use std::cell::{Cell, RefCell};
use std::rc::Rc;

thread_local! {
    static CLONE_GENERATION: Cell<u64> = const { Cell::new(0) };
}

/// Internal storage for a group of related references.
/// All `GraphRef`s created from the same `RefGraph` share the underlying data.
pub struct RefGraph<T> {
    data: RefCell<Vec<RefCell<T>>>,
    clone_cache: RefCell<Option<Rc<RefGraph<T>>>>,
    clone_generation: Cell<u64>,
}

impl<T> RefGraph<T> {
    /// Create a new empty RefGraph.
    pub fn new() -> Rc<Self> {
        Rc::new(RefGraph {
            data: RefCell::new(Vec::new()),
            clone_cache: RefCell::new(None),
            clone_generation: Cell::new(0),
        })
    }

    /// Create a new reference with the given initial value.
    pub fn create(self: &Rc<Self>, value: T) -> GraphRef<T> {
        let mut data = self.data.borrow_mut();
        let index = data.len();
        data.push(RefCell::new(value));
        GraphRef {
            graph: Rc::clone(self),
            index,
        }
    }

    /// Get the number of values stored in this graph.
    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }

    /// Check if this graph is empty.
    pub fn is_empty(&self) -> bool {
        self.data.borrow().is_empty()
    }

    /// Clear the clone cache to free memory.
    pub fn clear_cache(&self) {
        *self.clone_cache.borrow_mut() = None;
    }
}


impl<T: Clone> RefGraph<T> {
    /// Deep clone this graph, creating independent copies of all data.
    fn deep_clone_graph(self: &Rc<Self>) -> Rc<RefGraph<T>> {
        let data = self.data.borrow();
        let cloned_data: Vec<RefCell<T>> = data
            .iter()
            .map(|cell| RefCell::new(cell.borrow().clone()))
            .collect();

        Rc::new(RefGraph {
            data: RefCell::new(cloned_data),
            clone_cache: RefCell::new(None),
            clone_generation: Cell::new(0),
        })
    }

    /// Get or create a cached clone for the current generation.
    /// This is the key to fast deep cloning - O(1) lookup via generation check.
    #[inline]
    fn get_or_create_clone(self: &Rc<Self>, current_gen: u64) -> Rc<RefGraph<T>> {
        // Fast path: check if we already have a clone for this generation
        if self.clone_generation.get() == current_gen {
            // Clone exists for this generation
            if let Some(ref cached) = *self.clone_cache.borrow() {
                return Rc::clone(cached);
            }
        }

        // Slow path: create new clone
        let new_graph = self.deep_clone_graph();
        *self.clone_cache.borrow_mut() = Some(Rc::clone(&new_graph));
        self.clone_generation.set(current_gen);
        new_graph
    }
}

/// A cloneable reference that preserves structure during deep cloning.
///
/// - Regular `clone()` creates a shallow copy (same underlying data)
/// - When cloned during `deep_clone()`, creates structure-preserving deep copy
pub struct GraphRef<T> {
    graph: Rc<RefGraph<T>>,
    index: usize,
}

impl<T> GraphRef<T> {
    /// Get a copy of the referenced value.
    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.graph.data.borrow()[self.index].borrow().clone()
    }

    /// Set the referenced value.
    pub fn set(&self, value: T) {
        *self.graph.data.borrow()[self.index].borrow_mut() = value;
    }

    /// Apply a function to the referenced value.
    pub fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut T),
    {
        f(&mut *self.graph.data.borrow()[self.index].borrow_mut());
    }

    /// Get the index of this reference within its graph.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Check if two refs point to the same data (same graph and index).
    pub fn ptr_eq(&self, other: &GraphRef<T>) -> bool {
        Rc::ptr_eq(&self.graph, &other.graph) && self.index == other.index
    }

    /// Check if two refs are in the same graph (may have different indices).
    pub fn same_graph(&self, other: &GraphRef<T>) -> bool {
        Rc::ptr_eq(&self.graph, &other.graph)
    }
}

impl<T: Clone> Clone for GraphRef<T> {
    #[inline]
    fn clone(&self) -> Self {
        let current_gen = CLONE_GENERATION.with(|g| g.get());

        if current_gen > 0 {
            // We're in a deep clone operation - use generation-based lookup
            let new_graph = self.graph.get_or_create_clone(current_gen);
            GraphRef {
                graph: new_graph,
                index: self.index,
            }
        } else {
            // Shallow clone - same graph, same index
            GraphRef {
                graph: Rc::clone(&self.graph),
                index: self.index,
            }
        }
    }
}

impl<T: std::fmt::Debug + Clone> std::fmt::Debug for GraphRef<T> {
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
/// to the same (new) data after cloning.
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
    // Set up generation for this clone operation
    CLONE_GENERATION.with(|g| {
        let new_gen = g.get().wrapping_add(1).max(1); // Never 0 (0 means shallow clone)
        g.set(new_gen);
    });

    let result = value.clone();

    // Reset to shallow clone mode
    CLONE_GENERATION.with(|g| g.set(0));

    result
}

/// Guard that ensures clone generation is reset even on panic.
pub struct DeepCloneGuard {
    _private: (),
}

impl Drop for DeepCloneGuard {
    fn drop(&mut self) {
        CLONE_GENERATION.with(|g| g.set(0));
    }
}

/// Begin a deep clone operation manually. Returns a guard that resets state on drop.
///
/// Useful when you need more control over the cloning process.
/// Prefer `deep_clone()` for simple cases.
pub fn begin_deep_clone() -> DeepCloneGuard {
    CLONE_GENERATION.with(|g| {
        let new_gen = g.get().wrapping_add(1).max(1);
        g.set(new_gen);
    });
    DeepCloneGuard { _private: () }
}

// ============================================================================
// HashMap-based approach for comparison (slower)
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
        /// Pre-allocated array of new references
        ref_array: Vec<Rc<RefCell<T>>>,
        /// Cloned from PrepareNode
        pos_array: Vec<(bool, usize)>,
    }

    impl<T: Clone> PerformIndexCloningNode<T> {
        /// Create from a prepare node. This clones the pos_array.
        pub fn from_prepare_node(node: &PrepareIndexCloningNode<T>) -> Self {
            // Pre-allocate with dummy values (will be replaced)
            let mut ref_array = Vec::with_capacity(node.ref_pos_counter);
            ref_array.resize_with(node.ref_pos_counter, || Rc::new(RefCell::new(unsafe {
                // This is safe because we'll always set before reading
                std::mem::zeroed()
            })));

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
                self.ref_array[array_pos] = Rc::clone(&new_ref);
                IndexRef { inner: new_ref }
            } else {
                // Already cloned, return the cached one
                IndexRef {
                    inner: Rc::clone(&self.ref_array[array_pos]),
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
}
