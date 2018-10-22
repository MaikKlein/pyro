//! # What is an Entity Component System?
//! A game will have entities (gameobjects) in the world. Those entities can have a set
//! of components. A component is just data. Example for a component could be `Position`, `Velocity`, `Heath` etc.
//! Each entity can have a different set of components.
//!
//! The entiy component system can efficiently query those components.
//!
//! > Give me all entities that have a position and velocity component, and then update the position
//! based on the velocity.
//!
//! ```rust,ignore
//! type PosVelQuery = (Write<Pos>, Read<Vel>);
//! world.matcher::<All<PosVelQuery>>().for_each(|(pos, vel)|{
//!     pos += vel;
//! })
//! ```
//! # Internals
//!
//! ```ignore
//!// A Storage that contains `Pos`, `Vel`, `Health`.
//!(
//!    [Pos1, Pos2, Pos3, .., PosN],
//!    [Vel1, Vel2, Vel3, .., VelN],
//!    [Health1, Health2, Health3, .., HealthN],
//!)
//!
//!// A Storage that contains `Pos`, `Vel`.
//!(
//!    [Pos1, Pos2, Pos3, .., PosM]
//!    [Vel1, Vel2, Vel3, .., VelM]
//!)
//!
//! ```
//!
//! Iteration is fully linear with the exception of jumping to different storages.
//!
//! The iteration pattern from the query above would be
//!
//!
//! ```
//! positions:  [Pos1, Pos2, Pos3, .., PosN], [Pos1, Pos2, Pos3, .., PosM]
//! velocities: [Vel1, Vel2, Vel3, .., VelN], [Vel1, Vel2, Vel3, .., VelM]
//!                                         ^
//!                                         Jump occours here
//! ```
//! The jump is something like a chain of two iterators. We look at all the storages
//! that match specific query. If the query would be `Write<Position>`, then we would
//! look for all the storages that contain a position array, extract the iterators and chain them
//! together. [`Write`] in this content means that the iterator will be mutable, while
//! [`Read`] will create an immutable iterator.
//!
//!
//! Every combination of components will be a separate storage. This guarantees that iteration
//! will always be linear.
//!
//!
//! # Performance
//!
//! * **Iteration**: Extremely fast if each storage contains at least a few entities. For example
//! if a storage would contain only a single entity, then the performance wouldn't be better than a
//! linked list. At the moment there is only on storage `SoaStorage`. `SoA` is very good default
//! and you will most likely always want to use it. Different storages like `AOSOA` are planed. SoA
//! starts to decrease in performance when you iterate over a lot of different components.
//!
//! * **Insertion**: Very fast if you insert many entities at once. Entities can be added in an
//! AoS like fashion. This then gets translated to SoA which requires N allocations, where N is the
//! number of component types that you are inserting. Inserting single entities at a time is very
//! slow right now because the overhead of looking up the correct storages is very slow.
//!
//!* **Deletion**: Not implemented yet. Deletion in bulk will be fast while deletion of single
//!entities might be slow.
//!
//!* **Adding/Removing components**: Not yet implemented. Because iteration is always linear,
//!adding another component to an entity will move the entity to a different storage. This means
//!that all components will need to be moved into the new storage (shallow copy).
//!
//! # Benchmarks
//!
//! ![](https://raw.githubusercontent.com/MaikKlein/ecs_bench/master/graph/all.png)
//!
//! # Getting started
//!
extern crate itertools;
#[macro_use]
extern crate downcast_rs;
extern crate rayon;
use downcast_rs::Downcast;
use itertools::{multizip, Zip};
use rayon::prelude::*;
use std::any::TypeId;
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

pub type StorageId = u16;
pub type ComponentId = u32;

/// `Entitiy` serves as an ID to lookup components for entities which can be in
/// different storages.
// [TODO]: Make `Entity` generic.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Entity {
    /// Removing entities will increment the versioning. Accessing an entitiy with an
    /// out dated version will result in a `panic`.
    version: u16,
    /// The id of the storage where the entitiy lives in
    storage_id: StorageId,
    /// The actual id inside a storage
    id: ComponentId,
}

pub struct World<S = SoaStorage> {
    entities: Vec<Vec<Entity>>,
    storages: Vec<S>,
}

impl<S> World<S>
where
    S: Storage,
{
    pub fn matcher_with_entities<'s, Q>(
        &'s mut self,
    ) -> impl Iterator<Item = (Entity, <<Q as Query<'s>>::Iter as Iterator>::Item)> + 's
    where
        Q: Query<'s> + Matcher,
    {
        let entities = &self.entities;
        self.storages
            .iter()
            .enumerate()
            .filter(|&(_, storage)| Q::is_match(storage))
            // [FIXME] Honestly I am not sure why need to move the borrow from outside
            // into the closure. We get lifetime error for self otherwise, which is odd.
            .map(move |(storage_id, storage)| unsafe {
                let query = Q::query(storage);
                entities[storage_id].iter().cloned().zip(query)
            }).flat_map(|iter| iter)
    }
    pub fn matcher<'s, Q>(
        &'s mut self,
    ) -> impl Iterator<Item = <<Q as Query<'s>>::Iter as Iterator>::Item> + 's
    where
        Q: Query<'s> + Matcher,
    {
        unsafe {
            self.storages
                .iter()
                .filter(|&storage| Q::is_match(storage))
                .map(|storage| Q::query(storage))
                .flat_map(|iter| iter)
        }
    }
}
impl<S> World<S>
where
    S: Storage + RegisterComponent,
{
    pub fn new() -> Self {
        World {
            entities: Vec::new(),
            storages: Vec::new(),
        }
    }
    pub fn append_components<A, I>(&mut self, i: I)
    where
        A: AppendComponents + BuildStorage,
        I: IntoIterator<Item = A>,
    {
        // Try to find a matching storage, and insert the components
        let (storage_id, insert_count) = if let Some(storage) = self
            .storages
            .iter_mut()
            .find(|storage| A::is_match::<S>(storage))
        {
            let len = A::append_components(i, storage);
            (storage.id(), len)
        } else {
            // if we did not find a storage, we need to create one
            let id = self.storages.len() as StorageId;
            let mut storage = <A as BuildStorage>::build::<S>(id).access();
            let len = A::append_components(i, &mut storage);
            self.storages.push(storage);
            // Also we need to add an entity Vec for that storage
            self.entities.push(Vec::new());
            (id, len)
        };
        // Inserting components is not enough, we also need to create the entity ids
        // for those components.
        let start_len = self.entities[storage_id as usize].len() as u32;
        let end_len = start_len + insert_count as u32;
        let entities = (start_len..end_len).map(|id| Entity {
            storage_id: storage_id,
            id,
            version: 0,
        });
        self.entities[storage_id as usize].extend(entities);
    }
    pub fn remove_entities<I>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
    {

    }
}
pub trait Component: Send + 'static {}
impl<C: 'static + Send> Component for C {}

pub struct StorageBuilder<S: Storage> {
    current_id: ComponentId,
    storage_register: HashMap<ComponentId, S>,
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

    pub fn add_storage(&mut self, storage: S) -> ComponentId {
        let id = self.current_id + 1;
        self.storage_register.insert(id, storage);
        self.current_id = id;
        id
    }

    // pub fn extent_from_storage<C: Component>(&mut self, id: ComponentId) -> S {
    //     let mut s = self
    //         .storage_register
    //         .get(&id)
    //         .expect("Id not found")
    //         .clone();
    //     s.register_component::<C>();
    //     s
    // }
}

pub struct Exact<'s, Tuple>(pub PhantomData<&'s Tuple>);

impl<'s, A, B> Matcher for Exact<'s, (A, B)>
where
    A: Fetch<'s>,
    B: Fetch<'s>,
{
    fn is_match<S: Storage>(storage: &S) -> bool {
        let types = storage.types();
        types.len() == 2
            && types.contains(&TypeId::of::<A::Component>())
            && types.contains(&TypeId::of::<B::Component>())
    }
}

pub trait ReadComponent {
    type Component: Component;
}
pub trait WriteComponent {
    type Component: Component;
}
pub struct Read<C>(PhantomData<C>);
impl<C: Component> ReadComponent for Read<C> {
    type Component = C;
}

impl<C> Read<C> {
    pub fn new() -> Self {
        Read(PhantomData)
    }
}
impl<C: Component> WriteComponent for Write<C> {
    type Component = C;
}

pub struct Write<C>(PhantomData<C>);
pub trait Fetch<'s> {
    type Component: Component;
    type Iter: Iterator + 's;
    unsafe fn fetch<S: Storage>(storage: &'s S) -> Self::Iter;
}

impl<'s, C: Component> Fetch<'s> for Read<C> {
    type Component = C;
    type Iter = std::slice::Iter<'s, C>;
    unsafe fn fetch<S: Storage>(storage: &'s S) -> Self::Iter {
        storage.component::<C>().iter()
    }
}

impl<'s, C: Component> Fetch<'s> for Write<C> {
    type Component = C;
    type Iter = std::slice::IterMut<'s, C>;
    unsafe fn fetch<S: Storage>(storage: &'s S) -> Self::Iter {
        storage.component_mut::<C>().iter_mut()
    }
}

pub trait Matcher {
    fn is_match<S: Storage>(storage: &S) -> bool;
}
pub trait Query<'s> {
    type Iter: Iterator + 's;
    unsafe fn query<S: Storage>(storage: &'s S) -> Self::Iter;
}

pub struct All<'s, Tuple>(pub PhantomData<&'s Tuple>);
// impl<'s, A, B> Matcher for All<'s, (A, B)>
// where
//     A: Fetch<'s>,
//     B: Fetch<'s>,
// {
//     fn is_match<S: Storage>(storage: &S) -> bool {
//         storage.contains::<A::Component>() && storage.contains::<B::Component>()
//     }
// }
macro_rules! impl_matcher_all{
    ($($ty: ident),*) => {
        impl<'s, $($ty,)*> Matcher for All<'s, ($($ty,)*)>
        where
            $(
                $ty: Fetch<'s>,
            )*
        {
            fn is_match<S: Storage>(storage: &S) -> bool {
                $(
                    storage.contains::<$ty::Component>()
                )&&*
            }
        }
    }
}

impl_matcher_all!(A, B);
impl_matcher_all!(A, B, C);
impl_matcher_all!(A, B, C, D);
impl_matcher_all!(A, B, C, D, E);
impl_matcher_all!(A, B, C, D, E, F);
impl_matcher_all!(A, B, C, D, E, F, G);

impl<'s, A> Query<'s> for All<'s, A>
where
    A: Fetch<'s>,
{
    type Iter = A::Iter;
    unsafe fn query<S: Storage>(storage: &'s S) -> Self::Iter {
        A::fetch(storage)
    }
}

macro_rules! impl_query_all{
    ($($ty: ident),*) => {
        impl<'s, $($ty,)*> Query<'s> for All<'s, ($($ty,)*)>
        where
            $(
                $ty: Fetch<'s>,
            )*
        {
            type Iter = Zip<($($ty::Iter,)*)>;
            unsafe fn query<S: Storage>(storage: &'s S) -> Self::Iter {
                multizip(($($ty::fetch(storage),)*))
            }
        }
    }
}

impl_query_all!(A, B);
impl_query_all!(A, B, C);
impl_query_all!(A, B, C, D);
impl_query_all!(A, B, C, D, E);
impl_query_all!(A, B, C, D, E, F);
impl_query_all!(A, B, C, D, E, F, G);

pub struct EmptyStorage<S> {
    storage: S,
}

pub trait BuildStorage {
    fn build<S: Storage + RegisterComponent>(id: StorageId) -> EmptyStorage<S>;
}

macro_rules! impl_build_storage {
    ($($ty: ident),*) => {
        impl<$($ty),*> BuildStorage for ($($ty),*)
        where
            $(
                $ty:Component,
            )*
        {
            fn build<S: Storage + RegisterComponent>(id: StorageId) -> EmptyStorage<S> {
                let mut empty = S::empty(id);
                $(
                    empty.register_component::<$ty>();
                )*
                empty
            }
        }
    }
}
impl_build_storage!(A, B);
impl_build_storage!(A, B, C);
impl_build_storage!(A, B, C, D);
impl_build_storage!(A, B, C, D, E);
impl_build_storage!(A, B, C, D, E, F);

impl<S> EmptyStorage<S>
where
    S: Storage + RegisterComponent,
{
    pub fn register_component<C: Component>(&mut self) {
        self.storage.register_component::<C>();
    }
    pub fn access(self) -> S {
        self.storage
    }
}

pub trait RuntimeStorage: Downcast {
    fn remove(&mut self, id: ComponentId);
}

impl_downcast!(RuntimeStorage);
impl RuntimeStorage {
    pub fn as_unsafe_storage<C: Component>(&self) -> &UnsafeStorage<C> {
        self.downcast_ref::<UnsafeStorage<C>>()
            .expect("Incorrect storage type")
    }
    pub fn as_unsafe_storage_mut<C: Component>(&mut self) -> &mut UnsafeStorage<C> {
        self.downcast_mut::<UnsafeStorage<C>>()
            .expect("Incorrect storage type")
    }
}
impl<T: Component> RuntimeStorage for UnsafeStorage<T> {
    fn remove(&mut self, id: ComponentId) {}
}
// TODO: *Unsafe* Fix multiple mutable borrows
pub struct UnsafeStorage<T>(UnsafeCell<Vec<T>>);
impl<T> UnsafeStorage<T> {
    pub fn new() -> Self {
        UnsafeStorage(UnsafeCell::new(Vec::<T>::new()))
    }
    pub unsafe fn inner_mut(&mut self) -> &mut Vec<T> {
        &mut (*self.0.get())
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

pub trait ComponentList: Sized {
    const SIZE: usize;
    type Components;
}

macro_rules! impl_component_list {
    ($size: expr => $($ty: ident),*) => {
        impl<$($ty,)*> ComponentList for ($($ty),*) {
            const SIZE: usize = $size;
            type Components = ($($ty,)*);
        }
    }
}

impl_component_list!(2 => A, B);
impl_component_list!(3 => A, B, C);
impl_component_list!(4 => A, B, C, D);
impl_component_list!(5 => A, B, C, D, E);
impl_component_list!(6 => A, B, C, D, E, F);

pub trait IteratorSoa: Sized {
    type Output;
    fn to_soa<I: Iterator<Item = Self>>(iter: I) -> Self::Output;
}
macro_rules! impl_iterator_soa {
    ( $(($item: ident, $ty: ident )),*) => {
        impl<$($ty),*> IteratorSoa for ($($ty,)*)
        where
            $(
                $ty: Component,
            )*
        {
            type Output = ($(Vec<$ty>,)*);
            fn to_soa<I: Iterator<Item = Self>>(iter: I) -> Self::Output {
                $(
                    #[allow(non_snake_case)]
                    let mut $ty = Vec::new();
                )*
                for ($($item),*) in iter {
                    $(
                        $ty.push($item);
                    )*
                }
                ($($ty,)*)
            }
        }
    }
}

impl_iterator_soa!((a, A), (b, B));
impl_iterator_soa!((a, A), (b, B), (c, C));
impl_iterator_soa!((a, A), (b, B), (c, C), (d, D));
impl_iterator_soa!((a, A), (b, B), (c, C), (d, D), (e, E));
impl_iterator_soa!((a, A), (b, B), (c, C), (d, D), (e, E), (f, F));

pub trait AppendComponents
where
    Self: ComponentList + IteratorSoa,
{
    fn is_match<S: Storage>(storage: &S) -> bool;
    fn append_components<I, S>(items: I, storage: &mut S) -> usize
    where
        S: Storage,
        I: IntoIterator<Item = Self>;
}

macro_rules! impl_append_components {
    ($size: expr => $($ty: ident),*) => {
        impl<$($ty),*> AppendComponents for ($($ty,)*)
        where
            $(
                $ty: Component,
            )*
        {
            fn is_match<S: Storage>(storage: &S) -> bool {
                let types = storage.types();
                let mut b = types.len() == $size;
                $(
                    b &= types.contains(&TypeId::of::<$ty>());
                )*
                b
            }

            fn append_components<I, S>(items: I, storage: &mut S) -> usize
            where
                S: Storage,
                I: IntoIterator<Item = Self>,
            {
                let tuple = Self::to_soa(items.into_iter());
                let len = tuple.0.len();
                #[allow(non_snake_case)]
                let ($($ty,)*) = tuple;
                $(
                    storage.push_components($ty);
                )*
                len
            }
        }
    }
}

impl_append_components!(2 => A, B);
impl_append_components!(3 => A, B, C);
impl_append_components!(4 => A, B, C, D);
impl_append_components!(5 => A, B, C, D, E);
impl_append_components!(6 => A, B, C, D, E, F);

pub struct SoaStorage {
    len: usize,
    id: StorageId,
    types: HashSet<TypeId>,
    storages: HashMap<TypeId, Box<RuntimeStorage>>,
}

pub trait RegisterComponent {
    fn register_component<C: Component>(&mut self);
}

impl RegisterComponent for SoaStorage {
    fn register_component<C: Component>(&mut self) {
        let type_id = TypeId::of::<C>();
        self.types.insert(type_id);
        self.storages
            .insert(type_id, Box::new(UnsafeStorage::<C>::new()));
    }
}

// pub trait EntityIter<'a> {
//     type Iter: Iterator<Item = &'a Entity> + 'a;
//     fn entity_iter(&'a self) -> Self::Iter;
// }

// impl<'a> EntityIter<'a> for SoaStorage {
//     type Iter = std::slice::Iter<'a, Entity>;
//     fn entity_iter(&'a self) -> Self::Iter {
//         self.entities.iter()
//     }
// }

pub trait Storage: Sized {
    //fn create_entities(&mut self, count: u32);
    fn id(&self) -> StorageId;
    fn len(&self) -> usize;
    fn empty(id: StorageId) -> EmptyStorage<Self>;
    unsafe fn component<C: Component>(&self) -> &[C];
    unsafe fn component_mut<C: Component>(&self) -> &mut [C];
    fn push_components<C, I>(&mut self, components: I)
    where
        C: Component,
        I: IntoIterator<Item = C>;
    fn push_component<C: Component>(&mut self, component: C);
    fn contains<C: Component>(&self) -> bool;
    fn types(&self) -> &HashSet<TypeId>;
    fn remove(&mut self, id: ComponentId);
}

impl SoaStorage {
    fn get_storage<C: Component>(&self) -> &UnsafeStorage<C> {
        let runtime_storage = self
            .storages
            .get(&TypeId::of::<C>())
            .expect("Unknown storage");
        runtime_storage.as_unsafe_storage::<C>()
    }
    fn get_storage_mut<C: Component>(&mut self) -> &mut UnsafeStorage<C> {
        let runtime_storage = self
            .storages
            .get_mut(&TypeId::of::<C>())
            .expect("Unknown storage");
        runtime_storage.as_unsafe_storage_mut::<C>()
    }
}
impl Storage for SoaStorage {
    fn remove(&mut self, id: ComponentId) {
        self.types().iter().for_each(|type_id| {
            // let storage = self
            //     .anymap
            //     .get_mut::<UnsafeStorage<C>>()
            //     .expect("Component not found");
        });
    }
    fn id(&self) -> StorageId {
        self.id
    }
    fn len(&self) -> usize {
        self.len
    }
    fn push_components<C, I>(&mut self, components: I)
    where
        C: Component,
        I: IntoIterator<Item = C>,
    {
        unsafe {
            self.get_storage_mut::<C>().inner_mut().extend(components);
        }
    }
    fn push_component<C: Component>(&mut self, component: C) {
        self.get_storage::<C>().push(component);
    }
    fn empty(id: StorageId) -> EmptyStorage<Self> {
        let storage = SoaStorage {
            types: HashSet::new(),
            storages: HashMap::new(),
            id,
            len: 0,
        };
        EmptyStorage { storage }
    }
    unsafe fn component_mut<C: Component>(&self) -> &mut [C] {
        self.get_storage::<C>().get_mut_slice()
    }
    unsafe fn component<C: Component>(&self) -> &[C] {
        self.get_storage::<C>().get_slice()
    }

    fn contains<C: Component>(&self) -> bool {
        self.types.contains(&TypeId::of::<C>())
    }

    fn types(&self) -> &HashSet<TypeId> {
        &self.types
    }
}
// TODO: Implement a parallel multizip
// use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
// use std::cmp;
// #[derive(Debug, Clone)]
// pub struct MultiZip<Tuple> {
//     tuple: Tuple,
// }
// pub fn new<A, B>(a: A, b: B) -> MultiZip<(A, B)>
// where
//     A: IndexedParallelIterator,
//     B: IndexedParallelIterator,
// {
//     MultiZip { tuple: (a, b) }
// }

// impl<A, B> ParallelIterator for MultiZip<(A, B)>
// where
//     A: IndexedParallelIterator,
//     B: IndexedParallelIterator,
// {
//     type Item = (A::Item, B::Item);

//     fn drive_unindexed<C>(self, consumer: C) -> C::Result
//     where
//         C: UnindexedConsumer<Self::Item>,
//     {
//         bridge(self, consumer)
//     }

//     fn opt_len(&self) -> Option<usize> {
//         Some(self.len())
//     }
// }
// impl<A, B> IndexedParallelIterator for MultiZip<(A, B)>
// where
//     A: IndexedParallelIterator,
//     B: IndexedParallelIterator,
// {
//     fn drive<C>(self, consumer: C) -> C::Result
//     where
//         C: Consumer<Self::Item>,
//     {
//         bridge(self, consumer)
//     }

//     fn len(&self) -> usize {
//         cmp::min(self.tuple.0.len(), self.tuple.1.len())
//     }

//     fn with_producer<CB>(self, callback: CB) -> CB::Output
//     where
//         CB: ProducerCallback<Self::Item>,
//     {
//         return self.tuple.0.with_producer(CallbackA {
//             callback: callback,
//             b: self.tuple.1,
//         });

//         struct CallbackA<CB, B> {
//             callback: CB,
//             b: B,
//         }

//         impl<CB, ITEM, B> ProducerCallback<ITEM> for CallbackA<CB, B>
//         where
//             B: IndexedParallelIterator,
//             CB: ProducerCallback<(ITEM, B::Item)>,
//         {
//             type Output = CB::Output;

//             fn callback<A>(self, a_producer: A) -> Self::Output
//             where
//                 A: Producer<Item = ITEM>,
//             {
//                 return self.b.with_producer(CallbackB {
//                     a_producer: a_producer,
//                     callback: self.callback,
//                 });
//             }
//         }

//         struct CallbackB<CB, A> {
//             a_producer: A,
//             callback: CB,
//         }

//         impl<CB, A, ITEM> ProducerCallback<ITEM> for CallbackB<CB, A>
//         where
//             A: Producer,
//             CB: ProducerCallback<(A::Item, ITEM)>,
//         {
//             type Output = CB::Output;

//             fn callback<B>(self, b_producer: B) -> Self::Output
//             where
//                 B: Producer<Item = ITEM>,
//             {
//                 self.callback.callback(MultiZipProducer {
//                     tuple: (self.a_producer, b_producer),
//                 })
//             }
//         }
//     }
// }

// struct MultiZipProducer<Tuple> {
//     tuple: Tuple,
// }

// impl<A: Producer, B: Producer> Producer for MultiZipProducer<(A, B)> {
//     type Item = (A::Item, B::Item);
//     type IntoIter = itertools::structs::Zip<(A::IntoIter, B::IntoIter)>;

//     fn into_iter(self) -> Self::IntoIter {
//         multizip((self.tuple.0.into_iter(),self.tuple.1.into_iter()))
//     }

//     fn min_len(&self) -> usize {
//         cmp::max(self.tuple.0.min_len(), self.tuple.1.min_len())
//     }

//     fn max_len(&self) -> usize {
//         cmp::min(self.tuple.0.max_len(), self.tuple.1.max_len())
//     }

//     fn split_at(self, index: usize) -> (Self, Self) {
//         let (a_left, a_right) = self.tuple.0.split_at(index);
//         let (b_left, b_right) = self.tuple.1.split_at(index);
//         (
//             MultiZipProducer {
//                 tuple: (a_left, b_left),
//             },
//             MultiZipProducer {
//                 tuple: (a_right, b_right),
//             },
//         )
//     }
// }
