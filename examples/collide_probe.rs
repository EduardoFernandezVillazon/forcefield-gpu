use forcefield_gpu::{Backend, Config, ForceSpec, NodeInit, Simulation, Values};
use std::time::Instant;
fn overlaps(p: &[f32], r: f32) -> f32 {
    let n = p.len() / 2;
    let mut c = 0usize;
    for i in 0..n {
        for j in i + 1..n {
            let d = ((p[2 * i] - p[2 * j]).powi(2) + (p[2 * i + 1] - p[2 * j + 1]).powi(2)).sqrt();
            if d < 2.0 * r * 0.9 {
                c += 1
            }
        }
    }
    c as f32 / n as f32
}
fn main() {
    let n = 1500;
    let mut s = 11u64;
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
            links.push((a, b))
        }
    }
    let mut degree = vec![0u32; n];
    for &(a, b) in &links {
        degree[a as usize] += 1;
        degree[b as usize] += 1
    }
    let strength: Vec<f32> = links
        .iter()
        .map(|&(a, b)| 0.3 / degree[a as usize].min(degree[b as usize]).max(1) as f32)
        .collect();
    let cfg = |it: u32| {
        Config {
            alpha_decay: 0.015,
            velocity_decay: 0.35,
            ..Config::default()
        }
        .force(
            "collide",
            ForceSpec::Collide {
                radius: 25.0.into(),
                strength: 1.0,
                iterations: it,
            },
        )
        .force(
            "link",
            ForceSpec::Link {
                links: links.clone(),
                distance: 80.0.into(),
                strength: Some(Values::Each(strength.clone())),
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
        .force("center", ForceSpec::center(400.0, 300.0))
    };
    let mut cpu = Simulation::with_backend(Backend::Cpu, &nodes, &cfg(2)).unwrap();
    let t = Instant::now();
    cpu.run(400);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "cpu  it=2: overlaps/node {:.3}  ({ms:.0} ms)",
        overlaps(cpu.positions(), 25.0)
    );
    for it in [2u32, 3, 4, 6] {
        let mut gpu = Simulation::with_backend(Backend::Gpu, &nodes, &cfg(it)).unwrap();
        let t = Instant::now();
        gpu.run(400);
        let _ = gpu.positions();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "gpu  it={it}: overlaps/node {:.3}  ({ms:.0} ms)",
            overlaps(gpu.positions(), 25.0)
        );
    }
}
