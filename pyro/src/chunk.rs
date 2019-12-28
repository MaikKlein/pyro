use crate::{expand, Component, Slice, SliceMut};
use std::{
    alloc::{alloc, dealloc, realloc, Layout},
    any::TypeId,
    collections::{HashMap, HashSet},
    ptr::NonNull,
};

unsafe fn drop_generic<T>(ptr: *mut u8) {
    std::ptr::drop_in_place(ptr as *mut T);
}

#[derive(Copy, Clone)]
pub struct Metadata {
    layout: Layout,
    drop_fn: unsafe fn(ptr: *mut u8),
}
impl Metadata {
    pub fn of<C: Component>() -> Self {
        Self {
            layout: Layout::new::<C>(),
            drop_fn: drop_generic::<C>,
        }
    }
}

#[derive(Default, Clone)]
pub struct MetadataMap(HashMap<TypeId, Metadata>);
impl MetadataMap {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn insert<C: Component>(&mut self) {
        self.0.insert(TypeId::of::<C>(), Metadata::of::<C>());
    }
}


/// A runtime SoA storage. It stands for **S**tructure **o**f **A**rrays.
///
/// ```rust,ignore
/// struct Test {
///     foo: Foo,
///     bar: Bar,
///     baz: Baz,
/// }
/// let test: Vec<Test> = ...; // Array of Structs (*AoS*) layout
///
/// struct Test {
///     foo: Vec<Foo>,
///     bar: Vec<Bar>,
///     baz: Vec<Baz>,
/// }
/// let test: Test = ...; // SoA layout
// ```
/// Users do not interact with this storage directly, instead [`World`] will use this storage
/// behind the scenes. In the future there will be other storages such as *AoSoA*, which is a fancy
/// way of saying *SoA* but with arrays that a limited to a size of `8`.
pub struct Storage {
    types: HashSet<TypeId>,
    meta: MetadataMap,
    storages: HashMap<TypeId, NonNull<u8>>,
    // The maxiumum number of **elements** the storage can hold.
    capacity: usize,
    // The number of elements in the storate. `len <= capacity`
    len: usize,
}
unsafe impl Sync for Storage {}

impl Storage {
    pub fn meta(&self) -> &MetadataMap {
        &self.meta
    }
    pub fn contains<C: Component>(&self) -> bool {
        self.types.contains(&TypeId::of::<C>())
    }
    pub fn new(meta: MetadataMap) -> Self {
        Self {
            types: meta.0.keys().copied().collect(),
            meta,
            storages: HashMap::new(),
            capacity: 0,
            len: 0,
        }
    }

    pub fn components_raw<C: Component>(&self) -> Slice<C> {
        let allocation = self.storages.get(&TypeId::of::<C>()).unwrap();
        Slice::from_raw(allocation.as_ptr() as *const C, self.len())
    }

    pub fn components_mut_raw<C: Component>(&self) -> SliceMut<C> {
        let allocation = self
            .storages
            .get(&TypeId::of::<C>())
            .expect("Unknown type id");
        SliceMut::from_raw(allocation.as_ptr() as *mut C, self.len())
    }

    fn grow(&mut self) {
        if self.capacity == 0 {
            self.capacity = 1;
        } else {
            self.capacity *= 2;
        }
        let capacity = self.capacity;
        for (&id, &meta) in &self.meta.0 {
            let allocation = self
                .storages
                .entry(id)
                .or_insert_with(|| NonNull::new(unsafe { alloc(meta.layout) }).unwrap());

            if self.capacity > 1 {
                let size = meta.layout.size() * capacity;
                unsafe {
                    let new_allocation =
                        NonNull::new(realloc(allocation.as_ptr(), meta.layout, size)).unwrap();
                    *allocation = new_allocation;
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // TODO: Opt
    unsafe fn get_dyanmic_unchecked(&self, id: TypeId, idx: usize) -> *mut u8 {
        let size = self.meta.0.get(&id).unwrap().layout.size();
        let allocation = self.storages.get(&id).unwrap().as_ptr();
        allocation.add(size * idx)
    }

    pub fn swap_remove(&mut self, idx: usize) -> usize {
        assert!(!self.is_empty());
        let last_idx = self.len - 1;
        for (&id, meta) in &self.meta.0 {
            // First we swap the elements
            let ptr_to_delete = unsafe {
                let last_ptr = self.get_dyanmic_unchecked(id, last_idx);
                let cur_ptr = self.get_dyanmic_unchecked(id, idx);
                std::ptr::swap(last_ptr, cur_ptr);
                last_ptr
            };
            // Then we call drop on the last element
            unsafe {
                (meta.drop_fn)(ptr_to_delete);
            }
        }
        self.len -= 1;
        last_idx
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        for (ty, &meta) in &self.meta.0 {
            let allocation = self.storages.get(ty).unwrap();
            let size = meta.layout.size();
            // First we need to try to call drop on all the elements inside an allocation
            for idx in 0..self.len {
                unsafe {
                    let ptr = allocation.as_ptr().add(size * idx);
                    (meta.drop_fn)(ptr);
                }
            }
            unsafe {
                dealloc(allocation.as_ptr(), meta.layout);
            }
        }
    }
}

// [UNSOUND]: Unsound if not all components are initialized. Eg Storage with (A, B ,C), but extend
// is called with (A, B). This leaves C to be uninitialized, which is unsound in this abstraction
macro_rules! impl_extend {
    ($($ty: ident),*) => {
        impl<$($ty,)*> std::iter::Extend<($($ty,)*)> for Storage
        where
            $($ty: Component,)* {
            #[allow(non_snake_case)]
            fn extend<T>(&mut self, iter: T)
            where
                T: IntoIterator<Item = ($($ty,)*)> {
                let mut idx = self.len();
                for ($($ty,)*) in iter {
                    // TODO: Specialize on known length
                    if idx >= self.capacity {
                        self.grow();
                    }
                    //TODO Opt lookup
                    $(
                        unsafe {
                            let ptr = self.components_mut_raw::<$ty>().get_unchecked_mut(idx);
                            ptr.write($ty);
                        }
                    )*
                    idx +=1;
                }
                self.len = idx;
            }
        }
    }
}
expand!(impl_extend, A, B, C, D, E, F, G, H);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn swap_remove() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static mut DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);
        struct Int(u32);
        impl Drop for Int {
            fn drop(&mut self) {
                unsafe {
                    DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
        {
            let mut meta = MetadataMap::new();
            meta.insert::<u32>();
            meta.insert::<Int>();
            let mut storage = Storage::new(meta);
            storage.extend((0..10).map(|i| (i as u32, Int(i as _))));
            // We remove the first one which swaps it with the last one
            storage.swap_remove(0);
            println!("{:?}", storage.components::<u32>());
            let last: u32 = *storage.components().last().unwrap();
            assert_eq!(last, 8);
            let first: u32 = *storage.components().first().unwrap();
            assert_eq!(first, 9);
            unsafe { assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 1) }
        }
        unsafe { assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 10) }
    }
    #[test]
    fn storage_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static mut COUNTER: AtomicUsize = AtomicUsize::new(0);
        {
            let mut meta = MetadataMap::new();
            meta.insert::<Int>();
            meta.insert::<f32>();
            let mut storage = Storage::new(meta);
            struct Int(u32);
            impl Drop for Int {
                fn drop(&mut self) {
                    unsafe {
                        COUNTER.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
            storage.extend((0..10).map(|i| (Int(i), i as f32)));
        }
        unsafe {
            assert_eq!(COUNTER.load(Ordering::SeqCst), 10);
        }
    }
}
