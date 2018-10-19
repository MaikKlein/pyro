#![feature(test)]
extern crate ecs;
extern crate rand;
extern crate test;
use self::test::Bencher;
extern crate cgmath;
use cgmath::Vector2;
use ecs::*;
use rand::thread_rng;

type Vec2 = Vector2<f32>;
#[derive(Clone, Copy, Debug)]
struct Pos(Vec2);
#[derive(Clone, Copy, Debug)]
struct Vel(Vec2);
#[derive(Clone, Copy, Debug)]
struct Force(Vec2);
#[derive(Clone, Copy, Debug)]
struct InvMass(f32);
#[derive(Clone, Copy, Debug)]
struct Lifetime(f32);
#[derive(Clone, Copy, Debug)]
struct Ball {
    radius: f32,
}
#[derive(Clone, Copy, Debug)]
struct Rect {
    a: f32,
    b: f32,
}
#[derive(Clone, Copy, Debug)]
enum Spawner {
    Ball { radius: f32, inv_mass: f32 },
    Rect { a: f32, b: f32, inv_mass: f32 },
}
#[derive(Clone, Copy, Debug)]
struct SpawnRequests(usize);

#[derive(Clone, Copy, Debug)]
struct Room {
    inner_width: f32,
    inner_height: f32,
}
#[derive(Clone, Copy, Debug)]
enum Color {
    Green,
    Red,
}
#[derive(Clone, Copy, Debug, Default)]
struct KillsEnemy;
#[derive(Clone, Copy, Debug, Default)]
struct DeltaTime(f32);

fn integrate(world: &mut World, delta: DeltaTime) {
    use cgmath::Zero;
    let delta: f32 = delta.0;
    type IntegrateQuery = (Write<Pos>, Write<Vel>, Write<Force>, Read<InvMass>);
    world
        .matcher::<All<IntegrateQuery>>()
        .for_each(|(pos, vel, force, inv_mass)| {
            pos.0 += vel.0 * delta;

            let damping = (0.9f32).powf(delta);
            vel.0 += force.0 * inv_mass.0;
            vel.0 *= damping;

            force.0 = Vec2::zero();
        });
}

fn spawn(world: &mut World) {
    use cgmath::Zero;
    use rand::Rng;

    let mut rng = thread_rng();
    let mut gen = || rng.gen_range(-4.0, 4.0);

    let mut spawns = Vec::new();
    world
        .matcher::<All<(Read<Spawner>, Read<Pos>, Read<Color>, Write<SpawnRequests>)>>()
        .for_each(|(spawner, pos, color, requests)| {
            (0..requests.0).for_each(|_| {
                let spawn_pos = Vec2::new(gen(), gen());
                let spawn_pos = pos.0 + spawn_pos;
                spawns.push((*spawner, spawn_pos, *color));
            })
        });

    let rects = spawns
        .iter()
        .filter_map(|&(spawner, spawn_pos, spawn_color)| match spawner {
            Spawner::Rect { a, b, inv_mass } => {
                let rect = Rect { a, b };

                Some((
                    inv_mass,
                    Pos(spawn_pos),
                    Vel(Vec2::new(gen(), gen())),
                    Force(Vec2::zero()),
                    spawn_color,
                    rect,
                ))
            }
            _ => None,
        });
    world.add_entity(rects);
    let balls = spawns
        .iter()
        .filter_map(|&(spawner, spawn_pos, spawn_color)| match spawner {
            Spawner::Ball { radius, inv_mass } => Some((
                inv_mass,
                Pos(spawn_pos),
                Vel(Vec2::new(gen(), gen())),
                Force(Vec2::zero()),
                spawn_color,
                Ball { radius },
            )),
            _ => None,
        });
    world.add_entity(balls);
}

fn request_spawns(world: &mut World) {
    use rand::Rng;
    let mut rng = thread_rng();
    world
        .matcher::<All<Write<SpawnRequests>>>()
        .for_each(|requests| {
            let num = rng.gen_range(0, 200);
            if num > 197 {
                requests.0 = 2;
            }
        });
}

#[bench]
fn iter(b: &mut Bencher) {
    let mut world = World::<SoaStorage>::new();
    let delta = DeltaTime(0.02);
    for x in -50i32..50i32 {
        for y in -50i32..50i32 {
            let x = x as f32 * 35.0;
            let y = y as f32 * 30.0;
            let width = 30.0;
            let height = 25.0;

            let ball_spawner = Spawner::Ball {
                radius: 1.0,
                inv_mass: 2.0,
            };
            let rect_spawner = Spawner::Rect {
                a: 1.0,
                b: 3.0,
                inv_mass: 5.0,
            };

            let pos_x = [x - 8.0, x - 8.0, x + 8.0, x + 8.0];
            let pos_y = [y + 3.0, y - 3.0, y + 3.0, y - 3.0];
            let color = [Color::Green, Color::Green, Color::Red, Color::Red];
            let spawner = [ball_spawner, rect_spawner, ball_spawner, rect_spawner];

            let e = (0..1).map(|_| {
                (
                    Vec2::new(x, y),
                    Room {
                        inner_width: width,
                        inner_height: height,
                    },
                )
            });
            world.add_entity(e);

            let entities = (0..4).map(|i| {
                (
                    Pos(Vec2::new(pos_x[i], pos_y[i])),
                    spawner[i],
                    SpawnRequests(0),
                    Rect { a: 2.5, b: 2.5 },
                    color[i],
                )
            });
            world.add_entity(entities);
        }
    }
    b.iter(|| {
        request_spawns(&mut world);
        spawn(&mut world);
        integrate(&mut world, delta);
    });
}
