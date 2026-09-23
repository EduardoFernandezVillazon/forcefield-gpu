//! CPU vs GPU, consumer configuration, ms per tick.
//! cargo run --release --example bench
use forcefield_gpu::{probe, Backend, Config, ForceSpec, NodeInit, Simulation, Values};
use std::time::Instant;

fn main() {
    println!(
        "adapter: {:?}",
        probe().map(|a| format!("{} / {} / {}", a.name, a.backend, a.device_type))
    );
    println!("{:>7} {:>10} {:>10}", "nodes", "cpu ms", "gpu ms");
    for &n in &[2_000usize, 5_000, 10_000, 20_000, 50_000] {
        let mut s = 1u64;
        let mut rnd = || {
            s = (1664525 * s + 1013904223) % 4294967296;
            s as f32 / 4294967296.0
        };
        let nodes: Vec<NodeInit> = (0..n)
            .map(|_| NodeInit::at((rnd() * 800.0).round(), (rnd() * 600.0).round()))
            .collect();
        let mut links: Vec<(u32, u32)> = (1..n)
            .map(|i| ((rnd() * i as f32) as u32, i as u32))
            .collect();
        for _ in 0..n / 3 {
            let a = (rnd() * n as f32) as u32;
            let b = (rnd() * n as f32) as u32;
            if a != b {
                links.push((a, b));
            }
        }
        let mut degree = vec![0u32; n];
        for &(a, b) in &links {
            degree[a as usize] += 1;
            degree[b as usize] += 1;
        }
        let strength: Vec<f32> = links
            .iter()
            .map(|&(a, b)| 0.3 / degree[a as usize].min(degree[b as usize]).max(1) as f32)
            .collect();
        let config = Config {
            alpha_decay: 0.015,
            velocity_decay: 0.35,
            ..Config::default()
        }
        .force(
            "collide",
            ForceSpec::Collide {
                radius: 25.0.into(),
                strength: 1.0,
                iterations: 2,
            },
        )
        .force(
            "link",
            ForceSpec::Link {
                links,
                distance: 80.0.into(),
                strength: Some(Values::Each(strength)),
                iterations: 1,
            },
        )
        .force(
            "many-body",
            ForceSpec::ManyBody {
                strength: (-150.0).into(),
                theta: 0.9,
                distance_min: 5.0,
                distance_max: 250.0,
            },
        )
        .force(
            "x",
            ForceSpec::X {
                x: 0.0.into(),
                strength: 0.05.into(),
            },
        )
        .force(
            "y",
            ForceSpec::Y {
                y: 0.0.into(),
                strength: 0.05.into(),
            },
        )
        .force("center", ForceSpec::center(400.0, 300.0));
        let ticks = if n >= 20_000 { 20 } else { 50 };
        let cpu_ms = if n <= 20_000 {
            let mut sim = Simulation::with_backend(Backend::Cpu, &nodes, &config).unwrap();
            let t = Instant::now();
            sim.tick(ticks);
            t.elapsed().as_secs_f64() * 1e3 / ticks as f64
        } else {
            f64::NAN
        };
        let gpu_ms = match Simulation::with_backend(Backend::Gpu, &nodes, &config) {
            Ok(mut sim) => {
                sim.tick(2); // warm up pipelines
                let t = Instant::now();
                sim.tick(ticks);
                let _ = sim.positions(); // include the readback
                t.elapsed().as_secs_f64() * 1e3 / ticks as f64
            }
            Err(_) => f64::NAN,
        };
        println!("{:>7} {:>10.2} {:>10.2}", n, cpu_ms, gpu_ms);
    }
}
