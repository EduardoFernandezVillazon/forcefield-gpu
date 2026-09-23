# forcefield-gpu

The d3-force model of [forcefield](https://crates.io/crates/forcefield),
run as wgpu compute shaders, with the CPU crate as automatic fallback.
Same forces, same parameters, same alpha schedule, so a configuration
tuned against d3-force or forcefield carries over unchanged.

```rust
use forcefield_gpu::{Simulation, Config, ForceSpec, NodeInit};

let nodes = vec![NodeInit::UNSET; 20_000];             // NaN = phyllotaxis spiral, like d3
let config = Config { alpha_decay: 0.015, velocity_decay: 0.35, ..Config::default() }
    .force("collide", ForceSpec::Collide { radius: 25.0.into(), strength: 1.0, iterations: 2 })
    .force("link", ForceSpec::Link { links, distance: 80.0.into(), strength: None, iterations: 1 })
    .force("charge", ForceSpec::ManyBody { strength: (-150.0).into(), theta: 0.9, distance_min: 5.0, distance_max: 250.0 })
    .force("center", ForceSpec::center(0.0, 0.0));

let mut sim = Simulation::new(&nodes, &config);        // GPU if an adapter exists, else CPU
println!("{:?}", sim.backend());
sim.run(300);                                          // until alpha < alpha_min
let xy: &[f32] = sim.positions();                      // [x0, y0, x1, y1, …]
```

`Simulation::new` skips software adapters (llvmpipe, SwiftShader, WARP),
which are slower than the CPU crate; `Simulation::select(…, true)` allows
them. `Simulation::with_backend(Backend::Gpu, …)` forces the GPU and errors
without an adapter; `probe()` reports what would be used and whether it is
software. `GpuSimulation`
is the backend on its own, and `Gpu::acquire()` gives a device handle you
can share between simulations.

## What "same model" means here

Every force is the d3 formula with d3's parameters: many-body with
`distanceMin`/`distanceMax`, link with per-link distance, strength and
degree bias, collide with per-node radius and strength, x, y, radial,
center. Forces run in the configured order, then integration with
velocity decay and fixed nodes, exactly as d3 sequences them.

It is not bit-identical to forcefield, and cannot be:

- **float32** on the GPU against float64 on the CPU.
- **Many-body is exact pairwise** (theta 0) on the GPU; forcefield honours
  `theta` with Barnes-Hut. With `theta: 0.0` on both, positions agree to a
  relative 1e-3 after several ticks (tested).
- **Link and collide are gathered per node from a velocity snapshot**
  (Jacobi) where d3 updates pairs sequentially (Gauss-Seidel). Same
  displacements, different convergence per sweep. Collide gets one extra
  sweep on the GPU (`COLLIDE_EXTRA_SWEEPS`), which leaves fewer residual
  overlaps than the CPU at about 30 % more collide cost.

The tests check pass-for-pass agreement where the models coincide, layout
statistics at convergence for a real consumer configuration (edge length,
extent, nearest-neighbour distance within 10 %, overlaps no worse), pins,
determinism (same input, bit-identical output on one GPU), and the CPU
fallback.

## Speed

Consumer-shaped configuration (collide ×2, link, many-body with a distance
cap, x, y, center), ms per tick, one CPU core versus an integrated Intel
Arc GPU (Vulkan), 2026-09-23:

| nodes | forcefield (CPU) | forcefield-gpu |
|---:|---:|---:|
| 2 000 | 6.6 | 2.7 |
| 5 000 | 19.9 | 6.1 |
| 10 000 | 42.9 | 17.5 |
| 20 000 | 109.0 | 52.2 |
| 50 000 | | 183.9 |

`cargo run --release --example bench`. Repulsion and collision are brute
force O(n²) in workgroup-tiled compute, so this is the floor for an
integrated GPU, not the ceiling: a discrete GPU is several times faster,
and a grid or Barnes-Hut pass would change the complexity. The CPU crate is
O(n log n); above roughly 100 k nodes the brute-force GPU passes lose.

## Where it fits

- Native: one-shot layouts of large graphs in a Rust backend (for example a
  Tauri command), returning final coordinates.
- Browsers with WebGPU: the crate compiles to wasm; WebKitGTK (Tauri on
  Linux) has no WebGPU, so there the CPU wasm build stays in charge.

## Tests and CI

`cargo test -- --nocapture` prints the adapter and the layout statistics.
Without any adapter the GPU tests skip and only the CPU paths run. CI
installs Mesa's lavapipe so the GPU tests run on a runner without a GPU.

BSD-3-Clause. The model is d3-force's (Mike Bostock, BSD-3-Clause).
