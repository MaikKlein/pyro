extern crate cgmath;
extern crate ecs;
use cgmath::Vector2;
use ecs::*;

// type Vec2 = Vector2<f32>;
// #[derive(Clone, Copy, Debug)]
// struct Pos(Vec2);
// #[derive(Clone, Copy, Debug)]
// struct Vel(Vec2);
// #[derive(Clone, Copy, Debug)]
// struct Force(Vec2);
// #[derive(Clone, Copy, Debug)]
// struct InvMass(f32);
// #[derive(Clone, Copy, Debug)]
// struct Lifetime(f32);
// #[derive(Clone, Copy, Debug)]
// struct Ball {
//     radius: f32,
// }
// #[derive(Clone, Copy, Debug)]
// struct Rect {
//     a: f32,
//     b: f32,
// }
// #[derive(Clone, Copy, Debug)]
// enum Spawner {
//     Ball { radius: f32, inv_mass: f32 },
//     Rect { a: f32, b: f32, inv_mass: f32 },
// }
// #[derive(Clone, Copy, Debug)]
// struct SpawnRequests(usize);

// #[derive(Clone, Copy, Debug)]
// struct Collision {
//     a: Entity,
//     b: Entity,
//     contact: Vec2,
// }
// #[derive(Clone, Copy, Debug)]
// struct Room {
//     inner_width: f32,
//     inner_height: f32,
// }
// #[derive(Clone, Copy, Debug)]
// enum Color {
//     Green,
//     Red,
// }
// #[derive(Clone, Copy, Debug, Default)]
// struct KillsEnemy;
// #[derive(Clone, Copy, Debug, Default)]
// struct DeltaTime(f32);

