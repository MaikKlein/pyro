extern crate pyro;
use pyro::{Entity, Read, SoaStorage, World, Write};
struct Position;
struct Velocity;

fn main() {
    // By default creates a world backed by a [`SoaStorage`]
    let mut world: World<SoaStorage> = World::new();
    let add_pos_vel = (0..99).map(|_| (Position {}, Velocity {}));
    //                                 ^^^^^^^^^^^^^^^^^^^^^^^^
    //                                 A tuple of (Position, Velocity),
    //                                 Note: Order does *not* matter

    // Appends 99 entities with a Position and Velocity component.
    world.append_components(add_pos_vel);

    // Appends a single entity
    world.append_components(Some((Position {}, Velocity {})));

    // Requests a mutable borrow to Position, and an immutable borrow to Velocity.
    // Common queries can be reused with a typedef like this but it is not necessary.
    type PosVelQuery = (Write<Position>, Read<Velocity>);

    // Retrieves all entities that have a Position and Velocity component as an iterator.
    world.matcher::<PosVelQuery>().for_each(|(_pos, _vel)| {
        // ...
    });

    // The same query as above but also retrieves the entities and collects the entities into a
    // `Vec<Entity>`.
    let entities: Vec<Entity> = world
        .matcher_with_entities::<PosVelQuery>()
        .filter_map(|(entity, (_pos, _vel))| Some(entity))
        .collect();

    // Removes all the entities
    world.remove_entities(entities);
    let count = world.matcher::<PosVelQuery>().count();
    assert_eq!(count, 0);
}
