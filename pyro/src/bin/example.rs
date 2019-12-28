extern crate pyro;
use pyro::{Entity, Read, World, Write};
#[derive(Debug)]
struct Position(f32);
struct Velocity;

fn main() {
    // By default creates a world backed by a [`SoaStorage`]
    let mut world: World = World::new();
    let add_pos_vel = (0..10).map(|i| (Position(i as f32), Velocity {}));
    //                                 ^^^^^^^^^^^^^^^^^^^^^^^^
    //                                 A tuple of (Position, Velocity),
    //                                 Note: Order does *not* matter

    // Appends 99 entities with a Position and Velocity component.
    world.append_components(add_pos_vel);

    // Appends a single entity
    world.append_components(Some((Position(42.0), Velocity {})));

    // // Requests a mutable borrow to Position, and an immutable borrow to Velocity.
    // // Common queries can be reused with a typedef like this but it is not necessary.
    type PosVelQuery = (Write<Position>, Read<Velocity>);

    // Retrieves all entities that have a Position and Velocity component as an iterator.
    world
        .matcher::<(&mut Position, &Velocity)>()
        .for_each(|(pos, _vel)| {
            println!("{:?}", pos)
        });

    // The same query as above but also retrieves the entities and collects the entities into a
    // `Vec<Entity>`.
    let entities: Vec<Entity> = world
        .matcher_with_entities::<PosVelQuery>()
        .map(|(entity, _)| entity)
        .collect();

    // Removes all the entities
    world.remove_entities(entities);
    let count = world.matcher::<PosVelQuery>().count();
    assert_eq!(count, 0);
}
