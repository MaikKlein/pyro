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
//! world.matcher::<PosVelQuery>().for_each(|(pos, vel)|{
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
//!                                         Jump occurs here
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
//! use pyro::{ World, Entity, Read, Write};
//! struct Position;
//! struct Velocity;
//!
//!
//! let mut world: World = World::new();
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
//! world.matcher::<PosVelQuery>().for_each(|(pos, vel)|{
//!    // ...
//! });
//!
//! // The same query as above but also retrieves the entities and collects the entities into a
//! // `Vec<Entity>`.
//! let entities: Vec<Entity> =
//!     world.matcher_with_entities::<PosVelQuery>()
//!     .filter_map(|(entity, (pos, vel))|{
//!         Some(entity)
//!     }).collect();
//!
//! // Removes all the entities
//! world.remove_entities(entities);
//! let count = world.matcher::<PosVelQuery>().count();
//! assert_eq!(count, 0);
//! ```

mod chunk;
mod slice;
mod zip;
use chunk::{MetadataMap, Storage};
use log::debug;
use parking_lot::Mutex;
#[cfg(feature = "threading")]
pub use rayon::iter::{plumbing::UnindexedConsumer, IntoParallelRefIterator, ParallelIterator};
use slice::{Slice, SliceMut};
use std::{
    collections::HashSet,
    iter::{ExactSizeIterator, FusedIterator, IntoIterator},
    marker::PhantomData,
    num::Wrapping,
};
use typedef::TypeDef;
use vec_map::VecMap;
use zip::ZipSlice;

pub type StorageId = u16;
pub type ComponentId = u32;
pub type Version = u16;

pub trait Index<'a>: Sized {
    type Item;
    /// # Safety
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item;
    fn split_at(self, idx: usize) -> (Self, Self);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The [`Iterator`] is used to end a borrow from a query like [`World::matcher`].
pub struct BorrowIter<'s, I> {
    world: &'s World,
    iter: I,
}

#[cfg(feature = "threading")]
impl<'s, I> ParallelIterator for BorrowIter<'s, Option<I>>
where
    I: ParallelIterator,
{
    type Item = I::Item;
    fn drive_unindexed<C>(mut self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        self.iter.take().unwrap().drive_unindexed(consumer)
    }
}
impl<'s, I> Iterator for BorrowIter<'s, I>
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

impl<'s, I> FusedIterator for BorrowIter<'s, I> where I: FusedIterator {}

impl<'s, I> ExactSizeIterator for BorrowIter<'s, I>
where
    I: ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'s, I> Drop for BorrowIter<'s, I> {
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
pub struct World {
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
    storages: Vec<Storage>,
    /// The runtime borrow system. See [`RuntimeBorrow`] for more information. It is also wrapped
    /// in a Mutex so that we can keep track of multiple borrows on different threads.
    runtime_borrow: Mutex<RuntimeBorrow>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
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

impl World {
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
    #[cfg(feature = "threading")]
    pub fn par_matcher<'s, Q>(
        &'s self,
    ) -> impl ParallelIterator<Item = <<Q as ParQuery<'s>>::Iter as ParallelIterator>::Item> + 's
    where
        Q: ParQuery<'s> + Matcher,
        Q::Borrow: RegisterBorrow,
    {
        let iter = unsafe {
            self.storages
                .par_iter()
                .filter(|&storage| Q::is_match(storage))
                .map(|storage| Q::query(storage))
                .flat_map(|iter| iter)
        };
        BorrowIter {
            world: self,
            iter: Some(iter),
        }
    }
    /// Uses [`Query`] and [`Matcher`] to access the correct components. [`Read`] will borrow the
    /// component immutable while [`Write`] will borrow the component mutable.
    /// ```rust,ignore
    /// fn update(world: &mut World) {
    ///    world
    ///        .matcher::<(Write<Position>, Read<Velocity>)>()
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
                .flatten()
        };
        BorrowIter { world: self, iter }
    }
    /// Same as [`World::matcher`] but also returns the corresponding [`Entity`].
    /// ```rust,ignore
    /// fn update(world: &mut World) {
    ///    world
    ///        .matcher_with_entities::<(Write<Position>, Read<Velocity>)>()
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
        let iter = self
            .storages
            .iter()
            .enumerate()
            .filter(|&(_, storage)| Q::is_match(storage))
            .flat_map(move |(id, storage)| {
                let query = unsafe { Q::query(storage) };
                let entities = self.entities_storage(id as StorageId);
                Iterator::zip(entities, query)
            });
        BorrowIter { world: self, iter }
    }
}
impl World {
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
        let (storage_id, insert_count) = if let Some((id, storage)) = self
            .storages
            .iter_mut()
            .enumerate()
            .find(|(_, storage)| A::is_match(storage))
        {
            let len = A::append_components(i, storage);
            (id as StorageId, len)
        } else {
            // if we did not find a storage, we need to create one
            let id = self.storages.len() as StorageId;
            let mut storage = <A as BuildStorage>::build();
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

    /// Returns true if the entity owns the requested component.
    pub fn has_component<C: Component>(&self, e: Entity) -> bool {
        self.get_component::<C>(e).is_some()
    }

    /// Retrieves a component for a specific [`Entity`].
    pub fn get_component<C: Component>(&self, e: Entity) -> Option<&C> {
        let storage = &self.storages[e.storage_id as usize];
        if !storage.contains::<C>() || !self.is_entity_valid(e) {
            return None;
        }
        let component_id = self.component_map[e.storage_id as usize][e.id as usize];
        storage
            .components_raw::<C>()
            .try_get(component_id as usize)
            .map(|ptr| unsafe { &*ptr })
    }

    /// Same as [`World::get_component`] but mutable.
    // [TODO]: Possibly make this immutable and add the runtime borrow system if &mut isn't
    // flexible enough.
    pub fn get_component_mut<C: Component>(&mut self, e: Entity) -> Option<&mut C> {
        let storage = &self.storages[e.storage_id as usize];
        if !storage.contains::<C>() || !self.is_entity_valid(e) {
            return None;
        }
        let component_id = self.component_map[e.storage_id as usize][e.id as usize];
        storage
            .components_mut_raw::<C>()
            .try_get_mut(component_id as usize)
            .map(|ptr| unsafe { &mut *ptr })
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
            if !is_valid {
                continue;
            }
            let component_id = *self.component_map[storage_id]
                .get(entity.id as usize)
                .expect("component id");
            // [FIXME]: This uses dynamic dispatch so we might want to batch entities
            // together to reduce the overhead.
            let swap = self.storages[storage_id].swap_remove(component_id as _) as ComponentId;
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
            self.version[storage_id][entity.id as usize] += Wrapping(1);
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

impl<T: Component> PushBorrow for &'_ mut T {
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

impl<T: Component> PushBorrow for &'_ T {
    /// Multiple reads are always allowed and therefor we can always return true
    fn push_borrow(borrow: &mut Borrow) -> bool {
        borrow.reads.insert(TypeDef::of::<T>());
        true
    }
}

#[macro_export]
macro_rules! expand {
    ($m: ident, $ty: ident) => {
        $m!{$ty}
    };
    ($m: ident, $ty: ident, $($tt: ident),*) => {
        $m!{$ty, $($tt),*}
        expand!{$m, $($tt),*}
    };
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

expand!(impl_register_borrow, A, B, C, D, E, F, G, H);

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
                    })
                    .collect();
                reads.chain(rest)
            })
            .collect();
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
    type Iter: Index<'s>;
    /// # Safety
    unsafe fn fetch(storage: &'s Storage) -> Self::Iter;
}

impl<'s, C: Component> Fetch<'s> for Read<C> {
    type Component = C;
    type Iter = Slice<'s, C>;
    unsafe fn fetch(storage: &'s Storage) -> Self::Iter {
        storage.components_raw::<C>()
    }
}
impl<'s, C: Component> Fetch<'s> for &'_ C {
    type Component = C;
    type Iter = Slice<'s, C>;
    unsafe fn fetch(storage: &'s Storage) -> Self::Iter {
        storage.components_raw::<C>()
    }
}

impl<'s, C: Component> Fetch<'s> for Write<C> {
    type Component = C;
    type Iter = SliceMut<'s, C>;
    unsafe fn fetch(storage: &'s Storage) -> Self::Iter {
        storage.components_mut_raw::<C>()
    }
}
impl<'s, C: Component> Fetch<'s> for &'_ mut C {
    type Component = C;
    type Iter = SliceMut<'s, C>;
    unsafe fn fetch(storage: &'s Storage) -> Self::Iter {
        storage.components_mut_raw::<C>()
    }
}

/// Allows to match over different [`Storage`]s.
pub trait Matcher {
    fn is_match(storage: &Storage) -> bool;
}
/// Allows to query multiple components from a [`Storage`].
pub trait Query<'s> {
    type Borrow;
    type Iter: Iterator + 's;
    /// # Safety
    unsafe fn query(storage: &'s Storage) -> Self::Iter;
}
/// Allows to query multiple components from a [`Storage`] in parallel.
#[cfg(feature = "threading")]
pub trait ParQuery<'s> {
    type Borrow;
    type Iter: ParallelIterator + 's;
    unsafe fn query(storage: &'s Storage) -> Self::Iter;
}

impl<'a, T> Index<'a> for Slice<'a, T>
where
    T: 'a,
{
    type Item = &'a T;
    #[inline]
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item {
        &*self.start.add(idx)
    }
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
    fn split_at(self, idx: usize) -> (Self, Self) {
        Slice::split_at(self, idx)
    }
}
impl<'a, T> Index<'a> for SliceMut<'a, T>
where
    T: 'a,
{
    type Item = &'a mut T;
    #[inline]
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item {
        &mut *self.start.add(idx)
    }
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
    fn split_at(self, idx: usize) -> (Self, Self) {
        SliceMut::split_at_mut(self, idx)
    }
}

macro_rules! impl_matcher_default {
    ($($ty: ident),*) => {
        impl<$($ty,)*> Matcher for ($($ty,)*)
        where
            $(
                $ty: for<'s> Fetch<'s>,
            )*
        {
            fn is_match(storage: &Storage) -> bool {
                $(
                    storage.contains::<$ty::Component>()
                )&&*
            }
        }
        impl<'s, $($ty,)*> Query<'s> for ($($ty,)*)
        where
            $(
                $ty: Fetch<'s> + 's,
            )*
        {
            type Borrow = ($($ty,)*);
            type Iter = ZipSlice<'s, ($($ty::Iter,)*)>;
            unsafe fn query(storage: &'s Storage) -> Self::Iter {
                ZipSlice::new(($($ty::fetch(storage),)*))
            }
        }
        #[cfg(feature = "threading")]
        impl<'s, $($ty,)*> ParQuery<'s> for ($($ty,)*)
        where
            $(
                $ty: Fetch<'s> + Send + Sync + 's,
                <$ty as Fetch<'s>>::Iter: Send + Sync,
                <<$ty as Fetch<'s>>::Iter as Index<'s>>::Item: Send + Sync,
            )*
        {
            type Borrow = ($($ty,)*);
            type Iter = ZipSlice<'s, ($($ty::Iter,)*)>;
            unsafe fn query(storage: &'s Storage) -> Self::Iter {
                ZipSlice::new(($($ty::fetch(storage),)*))
            }
        }
    }
}

expand!(impl_matcher_default, A, B, C, D, E, F, G, H, I);

impl<'s, A> Query<'s> for A
where
    A: Fetch<'s> + 's,
{
    type Borrow = A;
    type Iter = ZipSlice<'s, (A::Iter,)>;
    unsafe fn query(storage: &'s Storage) -> Self::Iter {
        ZipSlice::new((A::fetch(storage),))
    }
}

impl<A> Matcher for A
where
    A: for<'s> Fetch<'s>,
{
    fn is_match(storage: &Storage) -> bool {
        storage.contains::<A::Component>()
    }
}

/// [`BuildStorage`] is used to create different [`Storage`]s at runtime. See also
/// [`AppendComponents`] and [`World::append_components`]
pub trait BuildStorage {
    fn build() -> Storage;
}

macro_rules! impl_build_storage {
    ($($ty: ident),*) => {
        impl<$($ty),*> BuildStorage for ($($ty,)*)
        where
            $(
                $ty:Component,
            )*
        {
            fn build() -> Storage {
                let mut meta = MetadataMap::new();
                $(
                    meta.insert::<$ty>();
                )*
                Storage::new(meta)
            }
        }
    }
}

expand!(impl_build_storage, A, B, C, D, E, F, G, H, I);

pub trait AppendComponents: Sized {
    fn is_match(storage: &Storage) -> bool;
    fn append_components<I>(items: I, storage: &mut Storage) -> usize
    where
        I: IntoIterator<Item = Self>;
}

macro_rules! impl_append_components {
    ($($ty: ident),*) => {
        impl<$($ty),*> AppendComponents for ($($ty,)*)
        where
            $(
                $ty: Component,
            )*
        {
            fn is_match(storage: &Storage) -> bool{
                $(
                    storage.contains::<$ty>()
                )&&*
            }
            fn append_components<Iter>(items: Iter, storage: &mut Storage) -> usize
            where
                Iter: IntoIterator<Item = Self>,
            {
                let mut count = 0;
                let iter = items.into_iter().map(|item| {
                    count += 1;
                    item
                });
                storage.extend(iter);
                count
            }
        }
    }
}

expand!(impl_append_components, A, B, C, D, E, F, G);
