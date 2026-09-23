//! One API over both backends: GPU when an adapter exists, CPU otherwise.

use crate::cpu;
use crate::gpu::{AdapterInfo, Gpu, GpuError, GpuSimulation};
use crate::spec::{Config, NodeInit};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu,
}

pub enum Simulation {
    Cpu {
        sim: Box<forcefield::Simulation>,
        positions: Vec<f32>,
    },
    Gpu(Box<GpuSimulation>),
}

impl Simulation {
    /// GPU if available, else CPU. Never fails.
    pub fn new(nodes: &[NodeInit], config: &Config) -> Simulation {
        match Gpu::acquire() {
            Ok(gpu) => match GpuSimulation::with_gpu(gpu, nodes, config) {
                Ok(s) => Simulation::Gpu(Box::new(s)),
                Err(_) => Simulation::cpu(nodes, config),
            },
            Err(_) => Simulation::cpu(nodes, config),
        }
    }

    /// A specific backend; `Gpu` errors when no adapter is present.
    pub fn with_backend(
        backend: Backend,
        nodes: &[NodeInit],
        config: &Config,
    ) -> Result<Simulation, GpuError> {
        Ok(match backend {
            Backend::Cpu => Simulation::cpu(nodes, config),
            Backend::Gpu => Simulation::Gpu(Box::new(GpuSimulation::new(nodes, config)?)),
        })
    }

    fn cpu(nodes: &[NodeInit], config: &Config) -> Simulation {
        Simulation::Cpu {
            sim: Box::new(cpu::build(nodes, config)),
            positions: vec![0.0; 2 * nodes.len()],
        }
    }

    pub fn backend(&self) -> Backend {
        match self {
            Simulation::Cpu { .. } => Backend::Cpu,
            Simulation::Gpu(_) => Backend::Gpu,
        }
    }

    pub fn adapter(&self) -> Option<&AdapterInfo> {
        match self {
            Simulation::Gpu(g) => Some(g.adapter()),
            _ => None,
        }
    }

    pub fn tick(&mut self, iterations: usize) {
        match self {
            Simulation::Cpu { sim, .. } => sim.tick(iterations),
            Simulation::Gpu(g) => g.tick(iterations),
        }
    }

    pub fn run(&mut self, max_ticks: usize) -> usize {
        match self {
            Simulation::Cpu { sim, .. } => sim.run(max_ticks),
            Simulation::Gpu(g) => g.run(max_ticks),
        }
    }

    pub fn positions(&mut self) -> &[f32] {
        match self {
            Simulation::Cpu { sim, positions } => {
                for (o, p) in positions.iter_mut().zip(sim.bodies().pos.iter()) {
                    *o = *p as f32;
                }
                positions
            }
            Simulation::Gpu(g) => g.positions(),
        }
    }

    pub fn alpha(&self) -> f32 {
        match self {
            Simulation::Cpu { sim, .. } => sim.alpha() as f32,
            Simulation::Gpu(g) => g.alpha(),
        }
    }
    pub fn set_alpha(&mut self, v: f32) {
        match self {
            Simulation::Cpu { sim, .. } => {
                sim.set_alpha(v as f64);
            }
            Simulation::Gpu(g) => {
                g.set_alpha(v);
            }
        }
    }
    pub fn alpha_min(&self) -> f32 {
        match self {
            Simulation::Cpu { sim, .. } => sim.alpha_min() as f32,
            Simulation::Gpu(g) => g.alpha_min(),
        }
    }
    pub fn set_alpha_target(&mut self, v: f32) {
        match self {
            Simulation::Cpu { sim, .. } => {
                sim.set_alpha_target(v as f64);
            }
            Simulation::Gpu(g) => {
                g.set_alpha_target(v);
            }
        }
    }
    pub fn set_alpha_decay(&mut self, v: f32) {
        match self {
            Simulation::Cpu { sim, .. } => {
                sim.set_alpha_decay(v as f64);
            }
            Simulation::Gpu(g) => {
                g.set_alpha_decay(v);
            }
        }
    }
    pub fn set_fixed(&mut self, i: usize, fx: f32, fy: f32) {
        match self {
            Simulation::Cpu { sim, .. } => sim.bodies_mut().set_fixed(i, fx as f64, fy as f64),
            Simulation::Gpu(g) => g.set_fixed(i, fx, fy),
        }
    }
}
