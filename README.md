# Pyro

A linear Entity Component System

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![LICENSE](https://img.shields.io/badge/license-apache-blue.svg)](LICENSE-APACHE)
[![Documentation](https://docs.rs/pyro/badge.svg)](https://docs.rs/pyro)
[![Crates.io Version](https://img.shields.io/crates/v/pyro.svg)](https://crates.io/crates/pyro)
[![](https://tokei.rs/b1/github/maikklein/pyro)](https://github.com/MaikKlein/pyro)

## Overview

Pyro is a tiny, fast and documented Entity Component System. It provides a basic features set as:

* Iterating over entities and components
* Adding and removing entities
* Tracks which handles are valid

The intention is to have a minimal set of features that can be built upon. 

## Implementation details

* Iteration is always **linear**.
* Different component combinations live in a separate storage
* Removing entities does not create holes.
* All operations are designed to be used in bulk.
* Borrow rules are enforced at runtime.
* `Entity` is using a wrapping generational index.

## Benchmarks

[bench defense](https://github.com/MaikKlein/bench_defense)

![](https://i.imgur.com/AyBFYAp.png)

[ecs_bench](https://github.com/MaikKlein/ecs_bench)

![](https://raw.githubusercontent.com/MaikKlein/ecs_bench/master/graph/all.png)


