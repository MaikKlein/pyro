extern crate ecs;
use ecs::*;
use std::marker::PhantomData;

type Query = (Read<u32>, Write<f32>);
fn main() {
    let mut world = World::<SoaStorage>::new();
    let entities = (0..100).map(|i|{
        (
            i as f32,
            i as u32
        )
    });
    world.add_entity(entities);
    let entities = (0..100).map(|i|{
        (
            "Hello",
            i as u32
        )
    });
    world.add_entity(entities);
    // // world.matcher::<All<Query>>().for_each(|(i, f)| {
    // //     println!("{}", f);
    // // });
    world.matcher::<All<(Read<u32>)>>().for_each(|(f)| {
        println!("{}", f);
    });
}
