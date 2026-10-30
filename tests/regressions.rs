//! Regressions found in review: a pass that ends inside `T::clone`, and deep clones in
//! thread-local destructors.

use graph_clonable_ref::{DeepCloneGuard, RefGraph, begin_deep_clone, deep_clone};
use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
    static HELD: RefCell<Option<DeepCloneGuard>> = const { RefCell::new(None) };
}

/// Ends the outermost pass while its graph's data is being cloned.
struct EndsPass;
impl Clone for EndsPass {
    fn clone(&self) -> Self {
        HELD.with(|h| drop(h.borrow_mut().take()));
        EndsPass
    }
}

#[test]
fn a_pass_that_ends_inside_t_clone_records_nothing() {
    let graph = RefGraph::new();
    let r = graph.create(EndsPass);
    let before = Arc::strong_count(&graph);

    HELD.with(|h| *h.borrow_mut() = Some(begin_deep_clone()));
    let first = r.clone(); // deep: the pass ends while the data is cloned
    assert!(!first.same_graph(&r));
    assert_eq!(
        Arc::strong_count(&graph),
        before,
        "original kept alive after the pass"
    );

    let second = deep_clone(&r);
    assert!(
        !second.same_graph(&first),
        "two passes share one clone graph"
    );
}

struct DeepClonesOnDrop;
impl Drop for DeepClonesOnDrop {
    fn drop(&mut self) {
        assert_eq!(deep_clone(&7u32), 7);
    }
}

thread_local! {
    static HOLDER: RefCell<Option<DeepClonesOnDrop>> = const { RefCell::new(None) };
}

#[test]
fn plain_deep_clone_in_a_thread_local_destructor_does_not_abort() {
    std::thread::spawn(|| {
        HOLDER.with(|h| *h.borrow_mut() = Some(DeepClonesOnDrop)); // destroyed after the map
        let _ = deep_clone(&RefGraph::new().create(1)); // creates the crate's thread-local map
    })
    .join()
    .unwrap();
}
