extern crate anymap;
extern crate itertools;
use anymap::{any::CloneAny, Map};
use itertools::{multizip, Zip};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::marker::PhantomData;

pub trait Component: 'static {}
impl<C: 'static> Component for C {}

pub trait BuildStorage {
    type Storage: Storage;
    fn build(&self) -> Self::Storage;
}

pub type StorageId = u32;
pub struct StorageBuilder<S: Storage> {
    current_id: StorageId,
    storage_register: HashMap<StorageId, S>,
}

impl<S> StorageBuilder<S>
where
    S: Storage,
{
    pub fn new() -> Self {
        StorageBuilder {
            current_id: 0,
            storage_register: HashMap::new(),
        }
    }

    pub fn add_storage(&mut self, storage: S) -> StorageId {
        let id = self.current_id + 1;
        self.storage_register.insert(id, storage);
        self.current_id = id;
        id
    }

    // pub fn extent_from_storage<C: Component>(&mut self, id: StorageId) -> S {
    //     let mut s = self
    //         .storage_register
    //         .get(&id)
    //         .expect("Id not found")
    //         .clone();
    //     s.register_component::<C>();
    //     s
    // }
}
pub trait Storage: Sized {
    fn empty() -> EmptyStorage<Self>;
    unsafe fn component<T: Component>(&self) -> Option<&[T]>;
    unsafe fn component_mut<T: Component>(&self) -> Option<&mut [T]>;
    fn append_components<I, A>(&mut self, components: I)
    where
        A: AppendComponents,
        I: IntoIterator<Item = A>;
    fn push_component<C: Component>(&mut self, component: C);
    fn contains<C: Component>(&self) -> bool;
}

pub struct All<'s, Tuple>(pub PhantomData<&'s Tuple>);

pub struct Read<C>(PhantomData<C>);
impl<C> Read<C> {
    pub fn new() -> Self {
        Read(PhantomData)
    }
}
pub struct Write<C>(PhantomData<C>);
pub trait Slice {
    fn len(&self) -> usize;
}
impl<T> Slice for &[T] {
    fn len(&self) -> usize {
        self.len()
    }
}
impl<T> Slice for &mut [T] {
    fn len(&self) -> usize {
        self.len()
    }
}
pub trait Fetch<'s> {
    type Component: Component;
    type Iter: Iterator;
    unsafe fn fetch<S: Storage>(storage: &'s S) -> Option<Self::Iter>;
}

// pub struct Iter<'s, C> {
//     idx: usize,
//     slice: &'s [C],
// }

// impl<'s, C> Iterator for Iter<'s, C> {
//     type Item = &'s C;
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.idx >= self.slice.len() {
//             return None;
//         }
//         let r = Some(&self.slice[self.idx]);
//         self.idx += 1;
//         r
//     }
// }

// pub struct IterMut<'s, C> {
//     idx: usize,
//     slice: &'s mut [C],
// }

// impl<'s, C> Iterator for IterMut<'s, C> {
//     type Item = &'s mut C;
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.idx >= self.slice.len() {
//             return None;
//         }
//         unsafe {
//             // [FIXME]: Why do I need to resort to unsafe here?
//             // What am I missing? Alternatively use two pointers instead of `&mut [C]`.
//             let ptr = self.slice.as_mut_ptr();
//             let r = ptr.add(self.idx).as_mut();
//             self.idx +=1;
//             r
//         }
//     }
// }

impl<'s, C: Component> Fetch<'s> for Read<C> {
    type Component = C;
    type Iter = std::slice::Iter<'s, C>;
    unsafe fn fetch<S: Storage>(storage: &'s S) -> Option<Self::Iter> {
        storage.component::<C>().map(|slice| slice.iter())
    }
}

impl<'s, C: Component> Fetch<'s> for Write<C> {
    type Component = C;
    type Iter = std::slice::IterMut<'s, C>;
    unsafe fn fetch<S: Storage>(storage: &'s S) -> Option<Self::Iter> {
        storage.component_mut::<C>().map(|slice| slice.iter_mut())
    }
}

pub trait Matcher<'s> {
    type Iter: Iterator;
    fn is_match<S: Storage>(storage: &'s mut S) -> bool;
    fn query<S: Storage>(storage: &'s mut S) -> Option<Self::Iter>;
}

impl<'s, A, B> Matcher<'s> for All<'s, (A, B)>
where
    A: Fetch<'s>,
    B: Fetch<'s>,
{
    type Iter = Zip<(A::Iter, B::Iter)>;
    fn is_match<S: Storage>(storage: &'s mut S) -> bool {
        storage.contains::<A::Component>() && storage.contains::<B::Component>()
    }
    fn query<S: Storage>(storage: &'s mut S) -> Option<Self::Iter> {
        unsafe {
            let i1 = A::fetch(storage)?;
            let i2 = B::fetch(storage)?;
            Some(multizip((i1, i2)))
        }
    }
}

pub struct EmptyStorage<S> {
    storage: S,
}

impl<S> EmptyStorage<S>
where
    S: Storage + Clone + RegisterComponent,
{
    pub fn register_component<C: Component>(&self) -> EmptyStorage<S> {
        let mut storage = self.storage.clone();
        storage.register_component::<C>();
        EmptyStorage { storage }
    }
    pub fn access(self) -> S {
        self.storage
    }
}

pub struct UnsafeStorage<T>(UnsafeCell<Vec<T>>);
impl<T> UnsafeStorage<T> {
    pub fn new() -> Self {
        UnsafeStorage(UnsafeCell::new(Vec::<T>::new()))
    }
    pub fn push(&self, t: T) {
        unsafe { (*self.0.get()).push(t) }
    }
    pub fn is_empty(&self) -> bool {
        unsafe { (*self.0.get()).is_empty() }
    }

    pub unsafe fn get_slice(&self) -> &[T] {
        (*self.0.get()).as_slice()
    }

    pub unsafe fn get_mut_slice(&self) -> &mut [T] {
        (*self.0.get()).as_mut_slice()
    }
}

impl<T> Clone for UnsafeStorage<T> {
    fn clone(&self) -> Self {
        assert!(self.is_empty());
        UnsafeStorage::new()
    }
}

pub trait AppendComponents: Sized {
    fn append_components<I, S>(items: I, storage: &mut S)
    where
        S: Storage,
        I: IntoIterator<Item = Self>;
}

impl<A, B> AppendComponents for (A, B)
where
    A: Component,
    B: Component,
{
    fn append_components<I, S>(items: I, storage: &mut S)
    where
        S: Storage,
        I: IntoIterator<Item = Self>,
    {
        for (a, b) in items {
            storage.push_component(a);
            storage.push_component(b);
        }
    }
}

#[derive(Clone)]
pub struct SoaStorage {
    anymap: Map<CloneAny>,
}

pub trait RegisterComponent {
    fn register_component<C: Component>(&mut self);
}

impl RegisterComponent for SoaStorage {
    fn register_component<C: Component>(&mut self) {
        self.anymap.insert(UnsafeStorage::<C>::new());
    }
}

impl Storage for SoaStorage {
    fn push_component<C: Component>(&mut self, component: C) {
        let storage = self
            .anymap
            .get_mut::<UnsafeStorage<C>>()
            .expect("Component not found");
        storage.push(component);
    }
    fn append_components<I, A>(&mut self, components: I)
    where
        A: AppendComponents,
        I: IntoIterator<Item = A>,
    {
        A::append_components(components, self);
    }
    fn empty() -> EmptyStorage<Self> {
        let storage = SoaStorage { anymap: Map::new() };
        EmptyStorage { storage }
    }
    unsafe fn component_mut<T: Component>(&self) -> Option<&mut [T]> {
        self.anymap
            .get::<UnsafeStorage<T>>()
            .map(|vec| vec.get_mut_slice())
    }
    unsafe fn component<T: Component>(&self) -> Option<&[T]> {
        self.anymap
            .get::<UnsafeStorage<T>>()
            .map(|vec| vec.get_slice())
    }

    fn contains<C: Component>(&self) -> bool {
        self.anymap.contains::<UnsafeStorage<C>>()
    }
}
