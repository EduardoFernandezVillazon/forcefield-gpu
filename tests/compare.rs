//! GPU backend against the CPU crate. Exact agreement is impossible (f32,
//! Jacobi gathers, different summation order), so the tests check three
//! things: pass-for-pass agreement where the models coincide, layout
//! statistics at convergence for the real consumer configuration, and
//! determinism.

use forcefield_gpu::{probe, Backend, Config, ForceSpec, NodeInit, Simulation, Values};

fn lcg(seed: u32) -> impl FnMut() -> f32 {
    let mut s = seed as u64;
    move || {
        s = (1664525 * s + 1013904223) % 4294967296;
        s as f32 / 4294967296.0
    }
}

fn graph(n: usize, extra: usize, seed: u32) -> (Vec<NodeInit>, Vec<(u32, u32)>) {
    let mut rnd = lcg(seed);
    let nodes = (0..n)
        .map(|_| NodeInit::at((rnd() * 800.0).round(), (rnd() * 600.0).round()))
        .collect();
    let mut links: Vec<(u32, u32)> = (1..n)
        .map(|i| ((rnd() * i as f32) as u32, i as u32))
        .collect();
    for _ in 0..extra {
        let a = (rnd() * n as f32) as u32;
        let b = (rnd() * n as f32) as u32;
        if a != b {
            links.push((a, b));
        }
    }
    (nodes, links)
}

fn consumer_config(links: Vec<(u32, u32)>, n: usize) -> Config {
    let mut degree = vec![0u32; n];
    for &(a, b) in &links {
        degree[a as usize] += 1;
        degree[b as usize] += 1;
    }
    let strength: Vec<f32> = links
        .iter()
        .map(|&(a, b)| 0.3 / degree[a as usize].min(degree[b as usize]).max(1) as f32)
        .collect();
    let mut c = Config {
        alpha_decay: 0.015,
        velocity_decay: 0.35,
        ..Config::default()
    };
    c = c
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
    c
}

fn gpu_available() -> bool {
    match probe() {
        Some(info) => {
            eprintln!(
                "GPU adapter: {} ({}, {})",
                info.name, info.backend, info.device_type
            );
            true
        }
        None => {
            eprintln!("no GPU adapter; GPU tests skipped");
            false
        }
    }
}

fn stats(pos: &[f32], links: &[(u32, u32)], radius: f32) -> (f32, f32, f32, f32) {
    let n = pos.len() / 2;
    let mean_edge = links
        .iter()
        .map(|&(a, b)| {
            ((pos[2 * a as usize] - pos[2 * b as usize]).powi(2)
                + (pos[2 * a as usize + 1] - pos[2 * b as usize + 1]).powi(2))
            .sqrt()
        })
        .sum::<f32>()
        / links.len() as f32;
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for i in 0..n {
        x0 = x0.min(pos[2 * i]);
        x1 = x1.max(pos[2 * i]);
        y0 = y0.min(pos[2 * i + 1]);
        y1 = y1.max(pos[2 * i + 1]);
    }
    let extent = ((x1 - x0) * (y1 - y0)).sqrt();
    let mut overlaps = 0usize;
    let mut nn = 0f32;
    for i in 0..n {
        let mut best = f32::MAX;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = ((pos[2 * i] - pos[2 * j]).powi(2) + (pos[2 * i + 1] - pos[2 * j + 1]).powi(2))
                .sqrt();
            if d < best {
                best = d;
            }
            if j > i && d < 2.0 * radius * 0.9 {
                overlaps += 1;
            }
        }
        nn += best;
    }
    (mean_edge, extent, overlaps as f32 / n as f32, nn / n as f32)
}

#[test]
fn many_body_and_positional_forces_agree_pass_for_pass() {
    if !gpu_available() {
        return;
    }
    // theta 0 makes the CPU many-body exact pairwise, like the GPU's brute force.
    let (nodes, _) = graph(400, 0, 3);
    let config = Config::default()
        .force(
            "charge",
            ForceSpec::ManyBody {
                strength: (-30.0).into(),
                theta: 0.0,
                distance_min: 1.0,
                distance_max: f32::INFINITY,
            },
        )
        .force(
            "x",
            ForceSpec::X {
                x: 100.0.into(),
                strength: 0.1.into(),
            },
        )
        .force(
            "y",
            ForceSpec::Y {
                y: (-50.0).into(),
                strength: 0.1.into(),
            },
        )
        .force(
            "radial",
            ForceSpec::Radial {
                radius: 200.0.into(),
                x: 0.0,
                y: 0.0,
                strength: 0.05.into(),
            },
        )
        .force("center", ForceSpec::center(10.0, 20.0));
    let mut cpu = Simulation::with_backend(Backend::Cpu, &nodes, &config).unwrap();
    let mut gpu = Simulation::with_backend(Backend::Gpu, &nodes, &config).unwrap();
    for t in 1..=5 {
        cpu.tick(1);
        gpu.tick(1);
        let a = cpu.positions().to_vec();
        let b = gpu.positions().to_vec();
        let mut worst = 0f32;
        for i in 0..a.len() {
            let scale = a[i].abs().max(100.0);
            worst = worst.max((a[i] - b[i]).abs() / scale);
        }
        eprintln!("tick {t}: worst relative deviation {worst:e}");
        assert!(worst < 1e-3, "tick {t}: GPU deviates from CPU by {worst:e}");
    }
    assert!((cpu.alpha() - gpu.alpha()).abs() < 1e-6);
}

#[test]
fn consumer_configuration_converges_to_an_equivalent_layout() {
    if !gpu_available() {
        return;
    }
    let n = 1500;
    let (nodes, links) = graph(n, n / 3, 11);
    let config = consumer_config(links.clone(), n);
    let mut cpu = Simulation::with_backend(Backend::Cpu, &nodes, &config).unwrap();
    let mut gpu = Simulation::with_backend(Backend::Gpu, &nodes, &config).unwrap();
    let tc = cpu.run(400);
    let tg = gpu.run(400);
    assert_eq!(tc, tg, "same alpha schedule → same tick count");
    let sc = stats(cpu.positions(), &links, 25.0);
    let sg = stats(gpu.positions(), &links, 25.0);
    eprintln!(
        "cpu: mean edge {:.1}, extent {:.0}, overlaps/node {:.3}, nn {:.1}",
        sc.0, sc.1, sc.2, sc.3
    );
    eprintln!(
        "gpu: mean edge {:.1}, extent {:.0}, overlaps/node {:.3}, nn {:.1}",
        sg.0, sg.1, sg.2, sg.3
    );
    let rel = |a: f32, b: f32| (a - b).abs() / a.abs().max(1e-6);
    assert!(
        rel(sc.0, sg.0) < 0.10,
        "mean edge length differs by {:.1}%",
        100.0 * rel(sc.0, sg.0)
    );
    assert!(
        rel(sc.1, sg.1) < 0.15,
        "layout extent differs by {:.1}%",
        100.0 * rel(sc.1, sg.1)
    );
    assert!(
        rel(sc.3, sg.3) < 0.10,
        "nearest-neighbour distance differs by {:.1}%",
        100.0 * rel(sc.3, sg.3)
    );
    assert!(
        sg.2 < sc.2 + 0.05,
        "GPU leaves more overlapping pairs ({:.3}/node vs {:.3})",
        sg.2,
        sc.2
    );
}

#[test]
fn pins_hold_and_runs_are_deterministic() {
    if !gpu_available() {
        return;
    }
    let (nodes, links) = graph(300, 50, 5);
    let config = consumer_config(links, 300);
    let mut a = Simulation::with_backend(Backend::Gpu, &nodes, &config).unwrap();
    let mut b = Simulation::with_backend(Backend::Gpu, &nodes, &config).unwrap();
    a.set_fixed(7, 123.0, -45.0);
    b.set_fixed(7, 123.0, -45.0);
    a.tick(50);
    b.tick(50);
    let pa = a.positions().to_vec();
    let pb = b.positions().to_vec();
    assert_eq!(pa[14], 123.0);
    assert_eq!(pa[15], -45.0);
    assert_eq!(
        pa, pb,
        "two GPU runs with the same input must be bit-identical"
    );
    a.set_fixed(7, f32::NAN, f32::NAN);
    a.tick(5);
    assert_ne!(a.positions()[14], 123.0, "unpinned node moves again");
}

#[test]
fn auto_selects_and_cpu_backend_always_works() {
    let (nodes, links) = graph(50, 10, 1);
    let config = consumer_config(links, 50);
    let mut sim = Simulation::new(&nodes, &config);
    eprintln!(
        "auto backend: {:?} {:?}",
        sim.backend(),
        sim.adapter().map(|a| a.name.clone())
    );
    sim.tick(3);
    assert!(sim.positions().iter().all(|v| v.is_finite()));
    let mut cpu = Simulation::with_backend(Backend::Cpu, &nodes, &config).unwrap();
    cpu.tick(3);
    assert_eq!(cpu.backend(), Backend::Cpu);
}
