extern crate ecs;
use ecs::*;
use std::marker::PhantomData;

type Query = (Read<u32>, Read<f32>);
fn main() {
    // let mut builder = StorageBuilder::<VecStorage>::new();
    let mut storage = SoaStorage::empty()
        .register_component::<f32>()
        .register_component::<u32>()
        .access();
    storage.append_components(vec![(1.0f32, 1u32)]);
    let r = All::<Query>::query(&mut storage).unwrap();
    println!("{:?}", r.collect::<Vec<_>>());
}
