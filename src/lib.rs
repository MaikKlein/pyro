extern crate anymap;
extern crate itertools;
use anymap::{any::CloneAny, Map};
use itertools::{multizip, Zip};
use std::any::TypeId;
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

pub struct World<S = SoaStorage> {
    storages: Vec<S>,
}

impl<S> World<S>
where
    S: Storage + RegisterComponent + Clone,
{
    pub fn new() -> Self {
        World {
            storages: Vec::new(),
        }
    }
    // [FIXME]: Why can't we use a impl trait here? An impl trait here results in lifetime issues.
    pub fn matcher<'s, Q>(&'s mut self) -> Box<Iterator<Item = <Q::Iter as Iterator>::Item> + 's>
    where
        Q: Query<'s>,
    {
        Box::new(
            self.storages
                .iter_mut()
                .filter_map(|storage| Q::query(storage))
                .flat_map(|iter| iter),
        )
    }
    pub fn add_entity<A, I>(&mut self, i: I)
    where
        A: AppendComponents,
        I: IntoIterator<Item = A>,
    {
        if let Some(storage) = self
            .storages
            .iter_mut()
            .find(|storage| A::is_match::<S>(storage))
        {
            A::append_components(i, storage);
        } else {
            let mut storage = <A as BuildStorage>::build::<S>().access();
            A::append_components(i, &mut storage);
            self.storages.push(storage);
        }
    }
}
pub trait Component: 'static {}
impl<C: 'static> Component for C {}

pub type StorageId = u32;
pub type ComponentId = u32;

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
pub trait Storage: Sized {
    fn empty() -> EmptyStorage<Self>;
    unsafe fn component<T: Component>(&self) -> Option<&[T]>;
    unsafe fn component_mut<T: Component>(&self) -> Option<&mut [T]>;
    fn push_components<C, I>(&mut self, components: I)
    where
        C: Component,
        I: IntoIterator<Item = C>;
    fn push_component<C: Component>(&mut self, component: C);
    fn contains<C: Component>(&self) -> bool;
    fn types(&self) -> &HashSet<TypeId>;
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

pub trait Matcher {
    fn is_match<S: Storage>(storage: &S) -> bool;
}
pub trait Query<'s> {
    type Iter: Iterator + 's;
    fn query<S: Storage>(storage: &'s mut S) -> Option<Self::Iter>;
}

pub struct All<'s, Tuple>(pub PhantomData<&'s Tuple>);
impl<'s, A, B> Matcher for All<'s, (A, B)>
where
    A: Fetch<'s>,
    B: Fetch<'s>,
{
    fn is_match<S: Storage>(storage: &S) -> bool {
        storage.contains::<A::Component>() && storage.contains::<B::Component>()
    }
}

impl<'s, A> Query<'s> for All<'s, A>
where
    A: Fetch<'s>,
{
    type Iter = A::Iter;
    fn query<S: Storage>(storage: &'s mut S) -> Option<Self::Iter> {
        unsafe { A::fetch(storage) }
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
            fn query<S: Storage>(storage: &'s mut S) -> Option<Self::Iter> {
                unsafe {
                    Some(multizip(($($ty::fetch(storage)?,)*)))
                }
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
    fn build<S: Storage + Clone + RegisterComponent>() -> EmptyStorage<S>;
}

macro_rules! impl_build_storage {
    ($($ty: ident),*) => {
        impl<$($ty),*> BuildStorage for ($($ty),*)
        where
            $(
                $ty:Component,
            )*
        {
            fn build<S: Storage + Clone + RegisterComponent>() -> EmptyStorage<S> {
                S::empty()$(.register_component::<$ty>())*
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

// TODO: *Unsafe* Fix multiple mutable borrows
pub struct UnsafeStorage<T>(UnsafeCell<Vec<T>>);
impl<T> UnsafeStorage<T> {
    pub fn new() -> Self {
        UnsafeStorage(UnsafeCell::new(Vec::<T>::new()))
    }
    pub unsafe fn inner_mut(&mut self) -> &mut Vec<T> {
        unsafe { &mut (*self.0.get()) }
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

pub trait ComponentList {
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

macro_rules! impl_append_components {
    ($size: expr => $($ty: ident),*) => {
        impl<$($ty),*> AppendComponents for ($($ty,)*)
        where
            $(
                $ty: Component,
            )*
        {
            //type ComponentList = ($($ty,)*);
            fn is_match<S: Storage>(storage: &S) -> bool {
                let types = storage.types();
                let mut b = types.len() == $size;
                $(
                    b &= types.contains(&TypeId::of::<$ty>());
                )*
                b
            }

            fn append_components<I, S>(items: I, storage: &mut S)
            where
                S: Storage,
                I: IntoIterator<Item = Self>,
            {
                #[allow(non_snake_case)]
                let ($($ty,)*) = Self::to_soa(items.into_iter());
                $(
                    storage.push_components($ty);
                )*
                // #[allow(non_snake_case)]
                // for ($($ty),*) in items {
                //     $(
                //         storage.push_component($ty);
                //     )*
                // }
            }
        }
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
            fn to_soa<I: Iterator<Item = Self>>(iter: I) -> Self::Output {
                $(
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
    Self: ComponentList + BuildStorage + Sized + IteratorSoa,
{
    fn is_match<S: Storage>(storage: &S) -> bool;
    fn append_components<I, S>(items: I, storage: &mut S)
    where
        S: Storage,
        I: IntoIterator<Item = Self>;
}

impl_append_components!(2 => A, B);
impl_append_components!(3 => A, B, C);
impl_append_components!(4 => A, B, C, D);
impl_append_components!(5 => A, B, C, D, E);
impl_append_components!(6 => A, B, C, D, E, F);

#[derive(Clone)]
pub struct SoaStorage {
    types: HashSet<TypeId>,
    anymap: Map<CloneAny>,
}

pub trait RegisterComponent {
    fn register_component<C: Component>(&mut self);
}

impl RegisterComponent for SoaStorage {
    fn register_component<C: Component>(&mut self) {
        self.types.insert(TypeId::of::<C>());
        self.anymap.insert(UnsafeStorage::<C>::new());
    }
}

impl Storage for SoaStorage {
    fn push_components<C, I>(&mut self, components: I)
    where
        C: Component,
        I: IntoIterator<Item = C>,
    {
        let storage = self
            .anymap
            .get_mut::<UnsafeStorage<C>>()
            .expect("Component not found");
        unsafe {
            storage.inner_mut().extend(components);
        }
    }
    fn push_component<C: Component>(&mut self, component: C) {
        let storage = self
            .anymap
            .get_mut::<UnsafeStorage<C>>()
            .expect("Component not found");
        storage.push(component);
    }
    fn empty() -> EmptyStorage<Self> {
        let storage = SoaStorage {
            types: HashSet::new(),
            anymap: Map::new(),
        };
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
    fn types(&self) -> &HashSet<TypeId> {
        &self.types
    }
}
