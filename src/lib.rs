//! # What is an Entity Component System?
//! An Entity Component System or *ECS* is very similar to a relational database like *SQL*. The
//! [`World`] is the data store where game objects (also known as [`Entity`]) live. An [`Entity`]
//! contains data or [`Component`]s.
//! The *ECS* can efficiently query those components.
//!
//! > Give me all entities that have a position and velocity component, and then update the position
//! based on the velocity.
//!
//! ```rust,ignore
//! type PosVelQuery = (Write<Pos>, Read<Vel>);
//! //                  ^^^^^       ^^^^
//! //                  Mutable     Immutable
//! world.matcher::<All<PosVelQuery>>().for_each(|(pos, vel)|{
//!     pos += vel;
//! })
//! ```
//!
//! # Internals
//! ## Overview
//! * Iteration is always **linear**.
//! * Different component combinations live in a separate storage
//! * Removing entities does not create holes.
//! * All operations are designed to be used in bulk.
//! * Borrow rules are enforced at runtime. See [`RuntimeBorrow`]
//! * [`Entity`] is using a wrapping generational index. See [`Entity::version`]
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
//! ```ignore
//! positions:  [Pos1, Pos2, Pos3, .., PosN], [Pos1, Pos2, Pos3, .., PosM]
//! velocities: [Vel1, Vel2, Vel3, .., VelN], [Vel1, Vel2, Vel3, .., VelM]
//!                                         ^
//!                                         Jump occours here
//! ```
//! The jump is something like a chain of two iterators. We look at all the storages
//! that match specific query. If the query would be `Write<Position>`, then we would
//! look for all the storages that contain a position array, extract the iterators and chain them
//!
//! Every combination of components will be in a separate storage. This guarantees that iteration
//! will always be linear.
//!
//! # Benchmarks
//!
//! ![](https://raw.githubusercontent.com/MaikKlein/ecs_bench/master/graph/all.png)
//!
//! # Getting started
//!
//! ```
//! extern crate pyro;
//! use pyro::{ World, Entity, Read, Write, All, SoaStorage };
//! struct Position;
//! struct Velocity;
//!
//!
//! // By default creates a world backed by a [`SoaStorage`]
//! let mut world: World<SoaStorage> = World::new();
//! let add_pos_vel = (0..99).map(|_| (Position{}, Velocity{}));
//! //                                 ^^^^^^^^^^^^^^^^^^^^^^^^
//! //                                 A tuple of (Position, Velocity),
//! //                                 Note: Order does *not* matter
//!
//! // Appends 99 entities with a Position and Velocity component.
//! world.append_components(add_pos_vel);
//!
//! // Appends a single entity
//! world.append_components(Some((Position{}, Velocity{})));
//!
//! // Requests a mutable borrow to Position, and an immutable borrow to Velocity.
//! // Common queries can be reused with a typedef like this but it is not necessary.
//! type PosVelQuery = (Write<Position>, Read<Velocity>);
//!
//! // Retrieves all entities that have a Position and Velocity component as an iterator.
//! world.matcher::<All<PosVelQuery>>().for_each(|(pos, vel)|{
//!    // ...
//! });
//!
//! // The same query as above but also retrieves the entities and collects the entities into a
//! // `Vec<Entity>`.
//! let entities: Vec<Entity> =
//!     world.matcher_with_entities::<All<PosVelQuery>>()
//!     .filter_map(|(entity, (pos, vel))|{
//!         Some(entity)
//!     }).collect();
//!
//! // Removes all the entities
//! world.remove_entities(entities);
//! let count = world.matcher::<All<PosVelQuery>>().count();
//! assert_eq!(count, 0);
//! ```
extern crate downcast_rs;
extern crate itertools;
extern crate log;
extern crate parking_lot;
extern crate rayon;
extern crate typedef;
extern crate vec_map;
use downcast_rs::{impl_downcast, Downcast};
use itertools::{multizip, Zip};
use log::debug;
use parking_lot::Mutex;
use std::any::TypeId;
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::num::Wrapping;
use typedef::TypeDef;
use vec_map::VecMap;

pub type StorageId = u16;
pub type ComponentId = u32;
pub type Version = u16;

/// The [`Iterator`] is used to end a borrow from a query like [`World::matcher`].
pub struct BorrowIter<'s, S, I> {
    world: &'s World<S>,
    iter: I,
}

impl<'s, S, I> Iterator for BorrowIter<'s, S, I>
where
    I: Iterator,
{
    type Item = I::Item;
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'s, S, I> FusedIterator for BorrowIter<'s, S, I> where I: FusedIterator {}

impl<'s, S, I> ExactSizeIterator for BorrowIter<'s, S, I>
where
    I: ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'s, S, I> Drop for BorrowIter<'s, S, I> {
    fn drop(&mut self) {
        self.world.runtime_borrow.lock().pop_access();
    }
}

/// Serves as an ID to lookup components for entities which can be in
/// different storages.
// [TODO]: Make `Entity` generic.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Entity {
    /// Removing entities will increment the versioning. Accessing an [`Entity`] with an
    /// outdated version will result in a `panic`. `version` does wrap on overflow.
    version: Wrapping<Version>,
    /// The id of the storage where the [`Entity`] lives in
    storage_id: StorageId,
    /// The actual id inside a storage
    id: ComponentId,
}

/// [`World`] is the heart of this library. It owns all the [`Component`]s and [`Storage`]s.
/// It also manages entities and allows [`Component`]s to be safely queried.
pub struct World<S = SoaStorage> {
    /// Storages need to be linear, that is why deletion will use [`Vec::swap_remove`] under the
    /// hood. But this moves the components around and we need to keep track of those swaps. This
    /// map is then used to find the correct [`ComponentId`] for an [`Entity`]. This maps the
    /// entity id to the real storage id.
    component_map: Vec<VecMap<ComponentId>>,
    /// This is the opposite of `component_map`. This maps the storage id to the entity id.
    component_map_inv: Vec<VecMap<ComponentId>>,
    /// When we remove an [`Entity`], we will put it in this free map to be reused.
    free_map: Vec<Vec<ComponentId>>,
    version: Vec<Vec<Wrapping<Version>>>,
    storages: Vec<S>,
    /// The runtime borrow system. See [`RuntimeBorrow`] for more information. It is also wrapped
    /// in a Mutex so that we can keep track of multiple borrows on different threads.
    runtime_borrow: Mutex<RuntimeBorrow>,
}

impl<S> Default for World<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> World<S> {
    /// Creates an empty [`World`].
    pub fn new() -> Self {
        World {
            runtime_borrow: Mutex::new(RuntimeBorrow::new()),
            component_map: Vec::new(),
            component_map_inv: Vec::new(),
            free_map: Vec::new(),
            version: Vec::new(),
            storages: Vec::new(),
        }
    }
}

impl<S> World<S>
where
    S: Storage,
{
    /// Creates an `Iterator` over every [`Entity`] inside [`World`]. The entities are
    /// not ordered.
    pub fn entities<'s>(&'s self) -> impl Iterator<Item = Entity> + 's {
        self.component_map
            .iter()
            .enumerate()
            .flat_map(move |(idx, inner)| {
                let storage_id = idx as StorageId;
                inner.keys().map(move |component_id| Entity {
                    storage_id,
                    id: component_id as ComponentId,
                    version: self.version[storage_id as usize][component_id as usize],
                })
            })
    }

    fn entities_storage<'s>(&'s self, storage_id: StorageId) -> impl Iterator<Item = Entity> + 's {
        // We iterate with the `component_map_inv`, because that is the order of the real storage.
        self.component_map_inv[storage_id as usize]
            .values()
            .map(move |&id| Entity {
                storage_id,
                id,
                version: self.version[storage_id as usize][id as usize],
            })
    }

    /// Pushes a new borrow on the stack and checks if there are any illegal overlapping borrows
    /// such as Write/Write and Read/Write.
    fn borrow_and_validate<Borrow: RegisterBorrow>(&self) {
        let mut borrow = self.runtime_borrow.lock();
        borrow.push_access::<Borrow>();
        // TODO: Implement a better error message.
        if let Err(overlapping_borrows) = borrow.validate() {
            panic!("Detected multiple active borrows of: {:?}", {
                overlapping_borrows
                    .iter()
                    .map(|ty| ty.get_str())
                    .collect::<Vec<_>>()
            });
        }
    }
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
        &'s self,
    ) -> impl Iterator<Item = <<Q as Query<'s>>::Iter as Iterator>::Item> + 's
    where
        Q: Query<'s> + Matcher,
        Q::Borrow: RegisterBorrow,
    {
        self.borrow_and_validate::<Q::Borrow>();
        let iter = unsafe {
            self.storages
                .iter()
                .filter(|&storage| Q::is_match(storage))
                .map(|storage| Q::query(storage))
                .flat_map(|iter| iter)
        };
        // [FIXME]: BorrowIter is more than 2x slower, than just returning `iter` here for the
        // `ecs_bench`. Maybe the benchmark is too simple?
        BorrowIter { world: self, iter }
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
        &'s self,
    ) -> impl Iterator<Item = (Entity, <<Q as Query<'s>>::Iter as Iterator>::Item)> + 's
    where
        Q: Query<'s> + Matcher,
        Q::Borrow: RegisterBorrow,
    {
        self.borrow_and_validate::<Q::Borrow>();
        // We need to explicitly tell rust how long we reborrow `self`, or the borrow checker
        // gets confused.
        let s: &'s Self = self;
        let iter = self
            .storages
            .iter()
            .enumerate()
            .filter(|&(_, storage)| Q::is_match(storage))
            .flat_map(move |(id, storage)| {
                let query = unsafe { Q::query(storage) };
                let entities = s.entities_storage(id as StorageId);
                Iterator::zip(entities, query)
            });
        BorrowIter { world: self, iter }
    }
}
impl<S> World<S>
where
    S: Storage + RegisterComponent,
{
    /// Appends the components and also creates the necessary [`Entity`]s behind the scenes.
    /// If you only want to append a single set of components then you can do
    /// ```rust,ignore
    /// world.append_components(Some((a, b, c)));
    /// ```
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
            self.component_map.push(VecMap::default());
            self.component_map_inv.push(VecMap::default());
            self.free_map.push(Vec::new());
            self.version.push(Vec::new());
            (id, len)
        };
        let storage_index = storage_id as usize;
        if insert_count == 0 {
            return;
        }
        // Inserting components is not enough, we also need to create the entity ids
        // for those components.
        let start_len = self.component_map[storage_index].len() as ComponentId;
        let end_len = start_len + insert_count as ComponentId;
        debug!("Append to Storage: {}", storage_id);
        debug!("- Insert count: {}", insert_count);
        debug!(
            "- Map length before: {}",
            self.component_map[storage_id as usize].len()
        );
        for component_id in start_len..end_len {
            if let Some(insert_at) = self.free_map[storage_index].pop() {
                // When we create a new entity that has already been deleted once, we need to
                // increment the version.
                self.version[storage_index][insert_at as usize] += Wrapping(1);
                self.insert_component_map(storage_id, insert_at, component_id);
            } else {
                // If the free list is empty, then we can insert it at the end.
                let insert_at = self.component_map[storage_index].len() as ComponentId;
                self.version[storage_index].push(Wrapping(0));
                self.insert_component_map(storage_id, insert_at, component_id);
            }
        }
        assert_eq!(
            self.component_map[storage_index].len(),
            self.storages[storage_index].len(),
            "The size of the component map and storage map should be equal"
        );
    }

    /// Compares the version of the entity with the version in [`World`] and returns true if they
    /// match. Because `version` wraps around this is not a hard guarantee.
    pub fn is_entity_valid(&self, entity: Entity) -> bool {
        self.version[entity.storage_id as usize]
            .get(entity.id as usize)
            .map(|&version| version == entity.version)
            .unwrap_or(false)
    }

    fn insert_component_map(
        &mut self,
        storage_id: StorageId,
        id: ComponentId,
        component_id: ComponentId,
    ) {
        self.component_map[storage_id as usize].insert(id as usize, component_id);
        self.component_map_inv[storage_id as usize].insert(component_id as usize, id);
    }

    /// Removes the specified entities from [`World`]. Those entities are now considered invalid,
    /// which can be checked with [`World::is_entity_valid`].
    pub fn remove_entities<I>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
    {
        for entity in entities {
            debug!("Removing {:?}", entity);
            let storage_id = entity.storage_id as usize;
            let is_valid = self.is_entity_valid(entity);
            assert!(is_valid, "Found an invalid Entitity");
            let component_id = self.component_map[storage_id][entity.id as usize];
            // [FIXME]: This uses dynamic dispatch so we might want to batch entities
            // together to reduce the overhead.
            let swap = self.storages[storage_id].remove(component_id) as ComponentId;
            // We need to keep track which entity was deleted and which was swapped.
            debug!(
                "- Entitiy id: {}, Component id: {}, Swap: {}, Map length: {}, Storage length: {}",
                entity.id,
                component_id,
                swap,
                self.storages[storage_id].len() + 1,
                self.component_map[storage_id].len()
            );

            // We need to look up the id that got swapped
            let key = self.component_map_inv[storage_id][swap as usize];
            debug!("- Updating {} to {}", key, component_id);
            // The id that was swapped should now point to the component_id that was removed
            self.insert_component_map(storage_id as StorageId, key, component_id);
            debug!("- Removing {} from `component_map`", entity.id);
            // Now we consider the id to be deleted and remove it from the `component_map`.
            self.component_map[storage_id as usize].remove(entity.id as usize);
            // We also need to update our `component_inverse_map`. `swap` was the real location
            // that was deleted in the underlying `storage` and we need to remove it.
            self.component_map_inv[storage_id as usize].remove(swap as usize);
            // And we need to append the remove id to the free map so we can reuse it again when we
            // `append_components`.
            self.free_map[storage_id].push(entity.id);
        }
    }
}

pub trait RegisterBorrow {
    /// Creates a new borrow
    fn register_borrow() -> Borrow;
}

/// Is implemented for [`Read`] and [`Write`] and is used to insert reads and writes into the
/// correct [`HashSet`].
pub trait PushBorrow {
    /// Inserts a new borrow and returns true if it was successful.
    fn push_borrow(acccess: &mut Borrow) -> bool;
}
impl<T: Component> PushBorrow for Write<T> {
    /// If a `Write` was already in a set, then we have detected multiple writes and this is not
    /// allows.
    fn push_borrow(borrow: &mut Borrow) -> bool {
        borrow.writes.insert(TypeDef::of::<T>())
    }
}

impl<T: Component> PushBorrow for Read<T> {
    /// Multiple reads are always allowed and therefor we can always return true
    fn push_borrow(borrow: &mut Borrow) -> bool {
        borrow.reads.insert(TypeDef::of::<T>());
        true
    }
}

macro_rules! impl_register_borrow{
    ($($ty: ident),*) => {
        impl<$($ty,)*> RegisterBorrow for ($($ty,)*)
        where
            $(
                $ty: PushBorrow,
            )*
        {
            fn register_borrow() -> Borrow {
                let mut borrow = Borrow::new();
                let success =
                $(
                    $ty::push_borrow(&mut borrow)
                )&&*;
                // TODO: Output a more meaningful error
                assert!(success, "Detected multiple writes");
                borrow
            }
        }
    }
}

impl_register_borrow!(A);
impl_register_borrow!(A, B);
impl_register_borrow!(A, B, C);
impl_register_borrow!(A, B, C, D);
impl_register_borrow!(A, B, C, D, E);
impl_register_borrow!(A, B, C, D, E, F);
impl_register_borrow!(A, B, C, D, E, F, G);
impl_register_borrow!(A, B, C, D, E, F, G, H);

/// Rust's borrowing rules are not flexible enough for an *ECS*. Often it would preferred to nest multiple
/// queries like [`World::matcher`], but this is not possible if both borrows would be mutable.
/// Instead we track active borrows at runtime. Multiple reads are allowed but `read/write` and
/// `write/write` are not.
pub struct RuntimeBorrow {
    borrows: Vec<Borrow>,
}

impl Default for RuntimeBorrow {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeBorrow {
    pub fn new() -> Self {
        Self {
            borrows: Vec::new(),
        }
    }

    /// Creates and pushes an [`Borrow`] on to the stack.
    pub fn push_access<R: RegisterBorrow>(&mut self) {
        let borrow = R::register_borrow();
        self.borrows.push(borrow);
    }
    /// Removes latest [`Borrow`]. This is usually called when an [`BorrowIter`] is dropped.
    pub fn pop_access(&mut self) {
        self.borrows.pop();
    }
    /// Validates the borrows. Multiple reads are allowed but Read/Write and Write/Write are not.
    pub fn validate(&self) -> Result<(), Vec<TypeDef>> {
        let overlapping_borrows: Vec<_> = self
            .borrows
            .iter()
            .enumerate()
            .flat_map(|(idx, borrow)| {
                let reads = borrow.writes.intersection(&borrow.reads).cloned();
                let rest: Vec<_> = self
                    .borrows
                    .iter()
                    .skip(idx + 1)
                    .flat_map(|next_access| {
                        let writes = borrow.writes.intersection(&next_access.writes).cloned();
                        let reads = borrow.writes.intersection(&next_access.reads).cloned();
                        writes.chain(reads)
                    }).collect();
                reads.chain(rest)
            }).collect();
        if overlapping_borrows.is_empty() {
            Ok(())
        } else {
            Err(overlapping_borrows)
        }
    }
}

pub struct Borrow {
    reads: HashSet<TypeDef>,
    writes: HashSet<TypeDef>,
}

impl Default for Borrow {
    fn default() -> Self {
        Self::new()
    }
}

impl Borrow {
    pub fn new() -> Self {
        Self {
            reads: HashSet::new(),
            writes: HashSet::new(),
        }
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
    type Iter: ExactSizeIterator + 's;
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
    type Borrow;
    type Iter: ExactSizeIterator + 's;
    unsafe fn query<S: Storage>(storage: &'s S) -> Self::Iter;
}

/// Is satisfied when a storages contains all of the specified components.
/// ```rust,ignore
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

impl_matcher_all!(A);
impl_matcher_all!(A, B);
impl_matcher_all!(A, B, C);
impl_matcher_all!(A, B, C, D);
impl_matcher_all!(A, B, C, D, E);
impl_matcher_all!(A, B, C, D, E, F);
impl_matcher_all!(A, B, C, D, E, F, G);
impl_matcher_all!(A, B, C, D, E, F, G, H);
// impl_matcher_all!(A, B, C, D, E, F, G, H, I);
// impl_matcher_all!(A, B, C, D, E, F, G, H, I, J);
// impl_matcher_all!(A, B, C, D, E, F, G, H, I, J, K);

impl<'s, A> Query<'s> for All<'s, A>
where
    A: Fetch<'s>,
{
    type Borrow = A;
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
            type Borrow = ($($ty,)*);
            type Iter = Zip<($($ty::Iter,)*)>;
            unsafe fn query<S1: Storage>(storage: &'s S1) -> Self::Iter {
                multizip(($($ty::fetch(storage),)*))
            }
        }
    }
}

impl_query_all!(A);
impl_query_all!(A, B);
impl_query_all!(A, B, C);
impl_query_all!(A, B, C, D);
impl_query_all!(A, B, C, D, E);
impl_query_all!(A, B, C, D, E, F);
impl_query_all!(A, B, C, D, E, F, G);
impl_query_all!(A, B, C, D, E, F, G, H);
// impl_query_all!(A, B, C, D, E, F, G, H, I);
// impl_query_all!(A, B, C, D, E, F, G, H, I, J);
// impl_query_all!(A, B, C, D, E, F, G, H, I, J, K);

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
        impl<$($ty),*> BuildStorage for ($($ty,)*)
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
impl_build_storage!(A);
impl_build_storage!(A, B);
impl_build_storage!(A, B, C);
impl_build_storage!(A, B, C, D);
impl_build_storage!(A, B, C, D, E);
impl_build_storage!(A, B, C, D, E, F);
impl_build_storage!(A, B, C, D, E, F, G);
impl_build_storage!(A, B, C, D, E, F, G, H);
// impl_build_storage!(A, B, C, D, E, F, G, H, I);
// impl_build_storage!(A, B, C, D, E, F, G, H, I, J);
// impl_build_storage!(A, B, C, D, E, F, G, H, I, J, k);

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
}

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
            fn to_soa<Iter: Iterator<Item = Self>>(iter: Iter) -> Self::Output {
                $(
                    #[allow(non_snake_case)]
                    let mut $ty = Vec::new();
                )*
                for ($($item,)*) in iter {
                    $(
                        $ty.push($item);
                    )*
                }
                ($($ty,)*)
            }
        }
    }
}

impl_iterator_soa!((a, A));
impl_iterator_soa!((a, A), (b, B));
impl_iterator_soa!((a, A), (b, B), (c, C));
impl_iterator_soa!((a, A), (b, B), (c, C), (d, D));
impl_iterator_soa!((a, A), (b, B), (c, C), (d, D), (e, E));
impl_iterator_soa!((a, A), (b, B), (c, C), (d, D), (e, E), (f, F));
impl_iterator_soa!((a, A), (b, B), (c, C), (d, D), (e, E), (f, F), (g, G));
impl_iterator_soa!(
    (a, A),
    (b, B),
    (c, C),
    (d, D),
    (e, E),
    (f, F),
    (g, G),
    (h, H)
);
impl_iterator_soa!(
    (a, A),
    (b, B),
    (c, C),
    (d, D),
    (e, E),
    (f, F),
    (g, G),
    (h, H),
    (i, I)
);
impl_iterator_soa!(
    (a, A),
    (b, B),
    (c, C),
    (d, D),
    (e, E),
    (f, F),
    (g, G),
    (h, H),
    (i, I),
    (j, J)
);
impl_iterator_soa!(
    (a, A),
    (b, B),
    (c, C),
    (d, D),
    (e, E),
    (f, F),
    (g, G),
    (h, H),
    (i, I),
    (j, J),
    (k, K)
);

pub trait AppendComponents
where
    Self: IteratorSoa,
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

            fn append_components<Iter, S>(items: Iter, storage: &mut S) -> usize
            where
                S: Storage,
                Iter: IntoIterator<Item = Self>,
            {
                let tuple = Self::to_soa(items.into_iter());
                let len = tuple.0.len();
                #[allow(non_snake_case)]
                let ($($ty,)*) = tuple;
                $(
                    storage.push_components($ty);
                )*
                *storage.len_mut() += len;
                len
            }
        }
    }
}

impl_append_components!(1  => A);
impl_append_components!(2  => A, B);
impl_append_components!(3  => A, B, C);
impl_append_components!(4  => A, B, C, D);
impl_append_components!(5  => A, B, C, D, E);
impl_append_components!(6  => A, B, C, D, E, F);
impl_append_components!(7  => A, B, C, D, E, F, G);
impl_append_components!(8  => A, B, C, D, E, F, G, H);
impl_append_components!(9  => A, B, C, D, E, F, G, H, I);
impl_append_components!(10 => A, B, C, D, E, F, G, H, I, J);
impl_append_components!(11 => A, B, C, D, E, F, G, H, I, J, K);

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

/// [`Storage`] allows to abstract over different types of storages. The most common storage that
/// implements this trait is [`SoaStorage`].
pub trait Storage: Sized {
    fn len(&self) -> usize;
    fn len_mut(&mut self) -> &mut usize;
    fn id(&self) -> StorageId;
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
    fn remove(&mut self, id: ComponentId) -> usize;
}

impl SoaStorage {
    /// Convenience method to easily access an [`UnsafeStorage`].
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
    fn len(&self) -> usize {
        self.len
    }
    fn len_mut(&mut self) -> &mut usize {
        &mut self.len
    }
    fn remove(&mut self, id: ComponentId) -> usize {
        self.storages.values_mut().for_each(|storage| {
            storage.remove(id);
        });
        self.len -= 1;
        self.len
    }
    fn id(&self) -> StorageId {
        self.id
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
