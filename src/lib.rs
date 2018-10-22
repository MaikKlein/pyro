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
use std::any::TypeId;
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

pub type StorageId = u16;
pub type ComponentId = u32;

/// Serves as an ID to lookup components for entities which can be in
/// different storages.
// [TODO]: Make `Entity` generic.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Entity {
    /// Removing entities will increment the versioning. Accessing an entitiy with an
    /// outdated version will result in a `panic`. `version` does wrap on overflow.
    version: u16,
    /// The id of the storage where the entitiy lives in
    storage_id: StorageId,
    /// The actual id inside a storage
    id: ComponentId,
}

/// [`World`] is the heart of this library. It owns all the [`Component`]s and [`Storage`]s.
/// It also manages entities and allows [`Component`]s to be safely queried.
pub struct World<S = SoaStorage> {
    entities: Vec<Vec<Entity>>,
    storages: Vec<S>,
}

impl<S> World<S>
where
    S: Storage,
{
    /// Uses [`Query`] and [`Matcher`] to access the correct components. [`Read`] will borrow the
    /// component immutable while [`Write`] will borrow the component mutable.
    /// ```rust,ignore
    /// fn update(world: &mut World) {
    ///    world
    ///        .matcher::<All<(Write<Position>, Read<Velocity>)>>()
    ///        .for_each(|(p, v)| {
    ///            p.x += v.dx;
    ///            p.y += v.dy;
    ///        });
    /// }
    /// ```
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
    /// Same as [`World::matcher`] but also returns the corresponding [`Entity`].
    /// ```rust,ignore
    /// fn update(world: &mut World) {
    ///    world
    ///        .matcher_with_entities::<All<(Write<Position>, Read<Velocity>)>>()
    ///        .for_each(|(entity, (p, v))| {
    ///            p.x += v.dx;
    ///            p.y += v.dy;
    ///        });
    /// }
    /// ```
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
}
impl<S> World<S>
where
    S: Storage + RegisterComponent,
{
    /// Creates an empty [`World`].
    pub fn new() -> Self {
        World {
            entities: Vec::new(),
            storages: Vec::new(),
        }
    }
    /// Appends the components and also creates the necessary [`Entity`]s behind the scenes.
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

/// Implements [`Fetch`] and allows components to be borrowed immutable.
pub struct Read<C>(PhantomData<C>);
/// Implements [`Fetch`] and allows components to be borrowed mutable.
pub struct Write<C>(PhantomData<C>);
/// A helper trait that works in lockstep with [`Read`] and [`Write`] to borrow components either
/// mutable or immutable.
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

/// Allows to match over different [`Storage`]s. See also [`All`].
pub trait Matcher {
    fn is_match<S: Storage>(storage: &S) -> bool;
}
/// Allows to query multiple components from a [`Storage`]. See also [`All`].
pub trait Query<'s> {
    type Iter: Iterator + 's;
    unsafe fn query<S: Storage>(storage: &'s S) -> Self::Iter;
}

/// Is satisfied when a storages contains all of the specified components.
/// ```rust,ingore
/// type Query = All<(Write<Position>, Read<Velocity>)>;
/// ```
pub struct All<'s, Tuple>(pub PhantomData<&'s Tuple>);

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

/// [`BuildStorage`] is used to create different [`Storage`]s at runtime. See also
/// [`AppendComponents`] and [`World::append_components`]
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
    fn remove(&mut self, id: ComponentId) {
        unsafe {
            self.inner_mut().swap_remove(id as usize);
        }
    }
}

// FIXME: *Unsafe* Fix multiple mutable borrows. Should be fixed in the `Query` API.
pub struct UnsafeStorage<T>(UnsafeCell<Vec<T>>);
impl<T> UnsafeStorage<T> {
    pub fn new() -> Self {
        UnsafeStorage(UnsafeCell::new(Vec::<T>::new()))
    }
    pub unsafe fn inner_mut(&self) -> &mut Vec<T> {
        &mut (*self.0.get())
    }
    // pub fn push(&self, t: T) {
    //     unsafe { (*self.0.get()).push(t) }
    // }
    // pub fn is_empty(&self) -> bool {
    //     unsafe { (*self.0.get()).is_empty() }
    // }

    // pub unsafe fn get_slice(&self) -> &[T] {
    //     (*self.0.get()).as_slice()
    // }

    // pub unsafe fn get_mut_slice(&self) -> &mut [T] {
    //     (*self.0.get()).as_mut_slice()
    // }
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
/// ```
/// Users do not interact with this storage directly, instead [`World`] will use this storage
/// behind the scenes. In the future there will be other storages such as *AoSoA*, which is a fancy
/// way of saying *SoA* but with arrays that a limited to a size of `8`.
pub struct SoaStorage {
    len: usize,
    id: StorageId,
    types: HashSet<TypeId>,
    storages: HashMap<TypeId, Box<RuntimeStorage>>,
}

/// A [`Storage`] won't have any arrays or vectors when it is created. [`RegisterComponent`] can
/// register or add those component arrays. See also [`EmptyStorage::register_component`]
pub trait RegisterComponent {
    fn register_component<C: Component>(&mut self);
}

impl RegisterComponent for SoaStorage {
    fn register_component<C: Component>(&mut self) {
        // A `SoAStorage` is backed by `UnsafeStorage`.
        let type_id = TypeId::of::<C>();
        self.types.insert(type_id);
        self.storages
            .insert(type_id, Box::new(UnsafeStorage::<C>::new()));
    }
}

/// [`Storage`] allows to abstract over differnt types of storages. The most common storage that
/// implements this trait is [`SoaStorage`].
pub trait Storage: Sized {
    fn id(&self) -> StorageId;
    fn len(&self) -> usize;
    /// Creates an [`EmptyStorage`]. This storage will not have any registered components when it
    /// is created. See [`RegisterComponent`].
    fn empty(id: StorageId) -> EmptyStorage<Self>;
    /// Retrieves the component array and panics if `C` is not inside this storage.
    unsafe fn component<C: Component>(&self) -> &[C];
    /// Same as [`Storage::component`] but mutable.
    unsafe fn component_mut<C: Component>(&self) -> &mut [C];
    /// Appends components to one array. See [`AppendComponents`] that uses this method.
    fn push_components<C, I>(&mut self, components: I)
    where
        C: Component,
        I: IntoIterator<Item = C>;
    fn push_component<C: Component>(&mut self, component: C);
    /// Returns true if the [`Storage`] has an array of type `C`.
    fn contains<C: Component>(&self) -> bool;
    fn types(&self) -> &HashSet<TypeId>;
    /// Removes **all** the components at the specified index.
    fn remove(&mut self, id: ComponentId);
}

impl SoaStorage {
    /// Convinence method to easily access an [`UnsafeStorage`].
    fn get_storage<C: Component>(&self) -> &UnsafeStorage<C> {
        let runtime_storage = self
            .storages
            .get(&TypeId::of::<C>())
            .expect("Unknown storage");
        runtime_storage.as_unsafe_storage::<C>()
    }
    /// Same as [`SoaStorage::get_storage`] but mutable.
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
        self.storages.values_mut().for_each(|storage| {
            storage.remove(id);
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
        unsafe {
            self.get_storage::<C>().inner_mut().push(component);
        }
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
        self.get_storage::<C>().inner_mut().as_mut_slice()
    }
    unsafe fn component<C: Component>(&self) -> &[C] {
        self.get_storage::<C>().inner_mut().as_slice()
    }

    fn contains<C: Component>(&self) -> bool {
        self.types.contains(&TypeId::of::<C>())
    }

    fn types(&self) -> &HashSet<TypeId> {
        &self.types
    }
}
