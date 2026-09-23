//! # forcefield-gpu
//!
//! The d3-force model of [forcefield](https://crates.io/crates/forcefield) on
//! the GPU through wgpu compute shaders, with the CPU crate as automatic
//! fallback. Same forces, same parameters, same alpha schedule; not
//! bit-identical (float32, and pairwise forces are gathered Jacobi-style
//! rather than applied sequentially), but the same layout.
//!
//! ```no_run
//! use forcefield_gpu::{Simulation, Config, ForceSpec, NodeInit};
//! let nodes = vec![NodeInit::UNSET; 10_000];
//! let config = Config::default()
//!     .force("charge", ForceSpec::many_body(-30.0))
//!     .force("link", ForceSpec::link(vec![(0, 1), (1, 2)]))
//!     .force("center", ForceSpec::center(0.0, 0.0));
//! let mut sim = Simulation::new(&nodes, &config); // GPU if present, else CPU
//! sim.run(300);
//! let xy = sim.positions();
//! ```

pub mod auto;
pub mod cpu;
pub mod gpu;
pub mod spec;

pub use auto::{Backend, Simulation};
pub use gpu::{probe, AdapterInfo, Gpu, GpuError, GpuSimulation};
pub use spec::{Config, ForceSpec, NodeInit, Values};
