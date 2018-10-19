#![feature(test)]
extern crate ecs;
extern crate test;
use self::test::Bencher;
use ecs::*;

#[derive(Debug, Copy, Clone)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct Velocity {
    pub x: f32,
    pub y: f32,
}

#[bench]
fn iter(b: &mut Bencher) {
    b.iter(|| {
        let mut world = World::<SoaStorage>::new();
        let entities = (0..100000).map(|i| {
            let pos = Position {
                x: i as f32,
                y: i as f32,
            };
            let vel = Velocity { x: 0.5, y: 0.5 };
            (pos, vel)
        });
        world.add_entity(entities);
        world
            .matcher::<All<(Write<Position>, Read<Velocity>)>>()
            .for_each(|(pos, vel)| {
                pos.x += vel.x;
                pos.y += vel.y;
            });
        world.matcher::<All<Read<Position>>>().fold(
            Position { x: 0.0, y: 0.0 },
            |mut acc, pos| {
                acc.x += pos.x;
                acc.y += pos.y;
                acc
            },
        )
    });
}
