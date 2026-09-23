//! The wgpu backend.

use crate::spec::{spiral, Config, ForceSpec, NodeInit, Values};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const WG: u32 = 256;
const NF: usize = 12;
const LF: usize = 3;
/// Extra collide sweeps run on the GPU to match d3's sequential collision pass.
pub const COLLIDE_EXTRA_SWEEPS: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no compatible GPU adapter found")]
    NoAdapter,
    #[error("device request failed: {0}")]
    Device(String),
    #[error("unsupported configuration: {0}")]
    Unsupported(String),
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Params {
    n: u32,
    m: u32,
    alpha: f32,
    velocity_decay: f32,
    distance_min2: f32,
    distance_max2: f32,
    collide_strength: f32,
    center_strength: f32,
    center_x: f32,
    center_y: f32,
    radial_x: f32,
    radial_y: f32,
    tick: u32,
    groups: u32,
    ptr_off: u32,
    idx_off: u32,
    ends_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[derive(Clone, Copy, Debug)]
enum Pass {
    ManyBody,
    Link { iterations: u32 },
    Collide { iterations: u32 },
    X,
    Y,
    Radial,
    Center,
}

/// A GPU adapter description, from [`probe`].
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    /// True for software rasterizers (llvmpipe, SwiftShader, WARP): valid
    /// adapters that run on the CPU, usually slower than the CPU crate.
    pub software: bool,
}

/// Shared device handle so several simulations can reuse one GPU device.
#[derive(Clone)]
pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub info: AdapterInfo,
}

async fn request_gpu() -> Result<Gpu, GpuError> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|_| GpuError::NoAdapter)?;
    let info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("forcefield-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            ..Default::default()
        })
        .await
        .map_err(|e| GpuError::Device(e.to_string()))?;
    Ok(Gpu {
        device: Arc::new(device),
        queue: Arc::new(queue),
        info: AdapterInfo {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            software: info.device_type == wgpu::DeviceType::Cpu,
        },
    })
}

impl Gpu {
    /// Acquire a device, blocking. `Err(NoAdapter)` means "use the CPU".
    pub fn acquire() -> Result<Gpu, GpuError> {
        pollster::block_on(request_gpu())
    }
}

/// Describe the adapter that would be used, without keeping a device.
pub fn probe() -> Option<AdapterInfo> {
    Gpu::acquire().ok().map(|g| g.info)
}

pub struct GpuSimulation {
    gpu: Gpu,
    n: u32,
    m: u32,
    params: Params,
    alpha_min: f32,
    alpha_decay: f32,
    alpha_target: f32,
    passes: Vec<Pass>,
    pipelines: Pipelines,
    bind_group: wgpu::BindGroup,
    buf_params: wgpu::Buffer,
    buf_pos: wgpu::Buffer,
    buf_vel: wgpu::Buffer,
    buf_vel_in: wgpu::Buffer,
    buf_nodes: wgpu::Buffer,
    buf_staging: wgpu::Buffer,
    nodes_f: Vec<f32>,
    nodes_dirty: bool,
    positions: Vec<f32>,
    positions_dirty: bool,
}

struct Pipelines {
    many_body: wgpu::ComputePipeline,
    link: wgpu::ComputePipeline,
    collide: wgpu::ComputePipeline,
    force_x: wgpu::ComputePipeline,
    force_y: wgpu::ComputePipeline,
    radial: wgpu::ComputePipeline,
    center_reduce1: wgpu::ComputePipeline,
    center_reduce2: wgpu::ComputePipeline,
    center_apply: wgpu::ComputePipeline,
    integrate: wgpu::ComputePipeline,
}

fn groups(n: u32) -> u32 {
    n.div_ceil(WG)
}

impl GpuSimulation {
    /// Build on a fresh device. Prefer [`GpuSimulation::with_gpu`] to share one.
    pub fn new(nodes: &[NodeInit], config: &Config) -> Result<Self, GpuError> {
        Self::with_gpu(Gpu::acquire()?, nodes, config)
    }

    pub fn with_gpu(gpu: Gpu, nodes: &[NodeInit], config: &Config) -> Result<Self, GpuError> {
        let n = nodes.len() as u32;
        let device = &gpu.device;

        // --- node state ---------------------------------------------------
        let mut pos = vec![0f32; 2 * nodes.len()];
        let mut nodes_f = vec![0f32; NF * nodes.len()];
        for (i, nd) in nodes.iter().enumerate() {
            let (mut x, mut y) = (nd.x, nd.y);
            if !nd.fx.is_nan() {
                x = nd.fx;
            }
            if !nd.fy.is_nan() {
                y = nd.fy;
            }
            if x.is_nan() || y.is_nan() {
                let (sx, sy) = spiral(i);
                x = sx;
                y = sy;
            }
            pos[2 * i] = x;
            pos[2 * i + 1] = y;
            let b = NF * i;
            nodes_f[b + 8] = if nd.fx.is_nan() { 0.0 } else { nd.fx };
            nodes_f[b + 9] = if nd.fy.is_nan() { 0.0 } else { nd.fy };
            nodes_f[b + 10] = if nd.fx.is_nan() { 0.0 } else { 1.0 };
            nodes_f[b + 11] = if nd.fy.is_nan() { 0.0 } else { 1.0 };
        }

        // --- forces → passes + packed parameters ------------------------
        let mut params = Params {
            n,
            velocity_decay: 1.0 - config.velocity_decay,
            distance_min2: 1.0,
            distance_max2: f32::INFINITY,
            collide_strength: 1.0,
            center_strength: 1.0,
            groups: groups(n),
            alpha: config.alpha,
            ..Default::default()
        };
        let mut passes = Vec::new();
        let mut links: Vec<(u32, u32)> = Vec::new();
        let mut links_f: Vec<f32> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let per_node = |v: &Values| v.to_vec(nodes.len());
        for (_, spec) in &config.forces {
            let kind = std::mem::discriminant(spec);
            if !seen.insert(kind) {
                return Err(GpuError::Unsupported(
                    "more than one force of the same kind".into(),
                ));
            }
            match spec {
                ForceSpec::ManyBody {
                    strength,
                    distance_min,
                    distance_max,
                    ..
                } => {
                    for (i, s) in per_node(strength).into_iter().enumerate() {
                        nodes_f[NF * i] = s;
                    }
                    params.distance_min2 = distance_min * distance_min;
                    params.distance_max2 = if distance_max.is_finite() {
                        distance_max * distance_max
                    } else {
                        f32::INFINITY
                    };
                    passes.push(Pass::ManyBody);
                }
                ForceSpec::Link {
                    links: ls,
                    distance,
                    strength,
                    iterations,
                } => {
                    let m = ls.len();
                    let mut count = vec![0u32; nodes.len()];
                    for &(s, t) in ls {
                        if s as usize >= nodes.len() || t as usize >= nodes.len() {
                            return Err(GpuError::Unsupported(format!(
                                "link endpoint out of range ({s}, {t})"
                            )));
                        }
                        count[s as usize] += 1;
                        count[t as usize] += 1;
                    }
                    let dist = distance.to_vec(m);
                    let strn: Vec<f32> = match strength {
                        Some(v) => v.to_vec(m),
                        None => ls
                            .iter()
                            .map(|&(s, t)| 1.0 / count[s as usize].min(count[t as usize]) as f32)
                            .collect(),
                    };
                    links_f = Vec::with_capacity(LF * m);
                    for (i, &(s, t)) in ls.iter().enumerate() {
                        links_f.push(dist[i]);
                        links_f.push(strn[i]);
                        links_f.push(
                            count[s as usize] as f32
                                / (count[s as usize] + count[t as usize]) as f32,
                        );
                    }
                    links = ls.clone();
                    passes.push(Pass::Link {
                        iterations: (*iterations).max(1),
                    });
                }
                ForceSpec::Collide {
                    radius,
                    strength,
                    iterations,
                } => {
                    for (i, r) in per_node(radius).into_iter().enumerate() {
                        nodes_f[NF * i + 1] = r;
                    }
                    params.collide_strength = *strength;
                    // The GPU gathers collisions Jacobi-style from a snapshot, which
                    // resolves overlaps more slowly per sweep than d3's sequential
                    // (Gauss-Seidel) pass. One extra sweep more than compensates
                    // (measured: fewer residual overlaps than the CPU at +30% collide cost).
                    passes.push(Pass::Collide {
                        iterations: (*iterations).max(1) + COLLIDE_EXTRA_SWEEPS,
                    });
                }
                ForceSpec::X { x, strength } => {
                    let xs = per_node(x);
                    let ss = per_node(strength);
                    for i in 0..nodes.len() {
                        nodes_f[NF * i + 2] = if xs[i].is_nan() { 0.0 } else { xs[i] };
                        nodes_f[NF * i + 3] = if xs[i].is_nan() { 0.0 } else { ss[i] };
                    }
                    passes.push(Pass::X);
                }
                ForceSpec::Y { y, strength } => {
                    let ys = per_node(y);
                    let ss = per_node(strength);
                    for i in 0..nodes.len() {
                        nodes_f[NF * i + 4] = if ys[i].is_nan() { 0.0 } else { ys[i] };
                        nodes_f[NF * i + 5] = if ys[i].is_nan() { 0.0 } else { ss[i] };
                    }
                    passes.push(Pass::Y);
                }
                ForceSpec::Center { x, y, strength } => {
                    params.center_x = *x;
                    params.center_y = *y;
                    params.center_strength = *strength;
                    passes.push(Pass::Center);
                }
                ForceSpec::Radial {
                    radius,
                    x,
                    y,
                    strength,
                } => {
                    let rs = per_node(radius);
                    let ss = per_node(strength);
                    for i in 0..nodes.len() {
                        nodes_f[NF * i + 6] = rs[i];
                        nodes_f[NF * i + 7] = if rs[i].is_nan() { 0.0 } else { ss[i] };
                    }
                    params.radial_x = *x;
                    params.radial_y = *y;
                    passes.push(Pass::Radial);
                }
            }
        }

        // --- CSR adjacency for the link gather ----------------------------
        let m = links.len() as u32;
        params.m = m;
        let mut ptr = vec![0u32; nodes.len() + 1];
        for &(s, t) in &links {
            ptr[s as usize + 1] += 1;
            ptr[t as usize + 1] += 1;
        }
        for i in 0..nodes.len() {
            ptr[i + 1] += ptr[i];
        }
        let mut fill = vec![0u32; nodes.len()];
        let mut idx = vec![0u32; 2 * links.len()];
        for (e, &(s, t)) in links.iter().enumerate() {
            for v in [s, t] {
                let v = v as usize;
                idx[(ptr[v] + fill[v]) as usize] = e as u32;
                fill[v] += 1;
            }
        }
        let mut links_u = Vec::with_capacity(ptr.len() + idx.len() + 2 * links.len());
        params.ptr_off = 0;
        links_u.extend_from_slice(&ptr);
        params.idx_off = links_u.len() as u32;
        links_u.extend_from_slice(&idx);
        params.ends_off = links_u.len() as u32;
        for &(s, t) in &links {
            links_u.push(s);
            links_u.push(t);
        }
        if links_u.is_empty() {
            links_u.push(0);
        }
        if links_f.is_empty() {
            links_f.push(0.0);
        }

        // --- buffers -------------------------------------------------------
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mk = |label: &str, data: &[u8], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage,
            })
        };
        let pos_bytes = (2 * nodes.len().max(1) * 4) as u64;
        let buf_params = mk(
            "params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let buf_pos = mk("pos", bytemuck::cast_slice(&pos), storage);
        let buf_vel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vel"),
            size: pos_bytes,
            usage: storage,
            mapped_at_creation: false,
        });
        let buf_vel_in = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vel_in"),
            size: pos_bytes,
            usage: storage,
            mapped_at_creation: false,
        });
        let buf_nodes = mk("nodes_f", bytemuck::cast_slice(&nodes_f), storage);
        let buf_links_u = mk(
            "links_u",
            bytemuck::cast_slice(&links_u),
            wgpu::BufferUsages::STORAGE,
        );
        let buf_links_f = mk(
            "links_f",
            bytemuck::cast_slice(&links_f),
            wgpu::BufferUsages::STORAGE,
        );
        let buf_reduce = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce"),
            size: (8 * groups(n).max(1)) as u64,
            usage: storage,
            mapped_at_creation: false,
        });
        let buf_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: pos_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- pipelines -----------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forcefield-gpu"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sim.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..8u32)
            .map(|b| wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: match b {
                        0 => wgpu::BufferBindingType::Uniform,
                        3..=6 => wgpu::BufferBindingType::Storage { read_only: true },
                        _ => wgpu::BufferBindingType::Storage { read_only: false },
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim"),
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            many_body: pipe("many_body"),
            link: pipe("link"),
            collide: pipe("collide"),
            force_x: pipe("force_x"),
            force_y: pipe("force_y"),
            radial: pipe("radial"),
            center_reduce1: pipe("center_reduce1"),
            center_reduce2: pipe("center_reduce2"),
            center_apply: pipe("center_apply"),
            integrate: pipe("integrate"),
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_vel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_vel_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_links_u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_links_f.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_reduce.as_entire_binding(),
                },
            ],
        });

        Ok(GpuSimulation {
            gpu,
            n,
            m,
            params,
            alpha_min: config.alpha_min,
            alpha_decay: config.alpha_decay,
            alpha_target: config.alpha_target,
            passes,
            pipelines,
            bind_group,
            buf_params,
            buf_pos,
            buf_vel,
            buf_vel_in,
            buf_nodes,
            buf_staging,
            nodes_f,
            nodes_dirty: false,
            positions: pos,
            positions_dirty: false,
        })
    }

    pub fn adapter(&self) -> &AdapterInfo {
        &self.gpu.info
    }
    pub fn len(&self) -> usize {
        self.n as usize
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn link_count(&self) -> usize {
        self.m as usize
    }

    // --- parameters --------------------------------------------------------
    pub fn alpha(&self) -> f32 {
        self.params.alpha
    }
    pub fn set_alpha(&mut self, v: f32) -> &mut Self {
        self.params.alpha = v;
        self
    }
    pub fn alpha_min(&self) -> f32 {
        self.alpha_min
    }
    pub fn set_alpha_min(&mut self, v: f32) -> &mut Self {
        self.alpha_min = v;
        self
    }
    pub fn alpha_decay(&self) -> f32 {
        self.alpha_decay
    }
    pub fn set_alpha_decay(&mut self, v: f32) -> &mut Self {
        self.alpha_decay = v;
        self
    }
    pub fn alpha_target(&self) -> f32 {
        self.alpha_target
    }
    pub fn set_alpha_target(&mut self, v: f32) -> &mut Self {
        self.alpha_target = v;
        self
    }
    pub fn velocity_decay(&self) -> f32 {
        1.0 - self.params.velocity_decay
    }
    pub fn set_velocity_decay(&mut self, v: f32) -> &mut Self {
        self.params.velocity_decay = 1.0 - v;
        self
    }

    /// Pin node `i` (NaN frees an axis).
    pub fn set_fixed(&mut self, i: usize, fx: f32, fy: f32) {
        let b = NF * i;
        self.nodes_f[b + 8] = if fx.is_nan() { 0.0 } else { fx };
        self.nodes_f[b + 9] = if fy.is_nan() { 0.0 } else { fy };
        self.nodes_f[b + 10] = if fx.is_nan() { 0.0 } else { 1.0 };
        self.nodes_f[b + 11] = if fy.is_nan() { 0.0 } else { 1.0 };
        self.nodes_dirty = true;
    }

    /// Overwrite a node's position (and zero its velocity on the next tick).
    pub fn set_position(&mut self, i: usize, x: f32, y: f32) {
        if self.positions_dirty {
            self.read_back();
        }
        self.positions[2 * i] = x;
        self.positions[2 * i + 1] = y;
        self.gpu.queue.write_buffer(
            &self.buf_pos,
            (8 * i) as u64,
            bytemuck::cast_slice(&self.positions[2 * i..2 * i + 2]),
        );
    }

    // --- stepping ------------------------------------------------------------

    pub fn tick(&mut self, iterations: usize) {
        if self.n == 0 {
            return;
        }
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        if self.nodes_dirty {
            queue.write_buffer(&self.buf_nodes, 0, bytemuck::cast_slice(&self.nodes_f));
            self.nodes_dirty = false;
        }
        let g = groups(self.n);
        let vel_bytes = (8 * self.n) as u64;
        for _ in 0..iterations {
            self.params.alpha += (self.alpha_target - self.params.alpha) * self.alpha_decay;
            self.params.tick = self.params.tick.wrapping_add(1);
            queue.write_buffer(&self.buf_params, 0, bytemuck::bytes_of(&self.params));
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tick"),
            });
            let dispatch =
                |enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, groups: u32| {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(groups, 1, 1);
                };
            for p in &self.passes {
                match *p {
                    Pass::ManyBody => dispatch(&mut enc, &self.pipelines.many_body, g),
                    Pass::Link { iterations } => {
                        for _ in 0..iterations {
                            enc.copy_buffer_to_buffer(
                                &self.buf_vel,
                                0,
                                &self.buf_vel_in,
                                0,
                                vel_bytes,
                            );
                            dispatch(&mut enc, &self.pipelines.link, g);
                        }
                    }
                    Pass::Collide { iterations } => {
                        for _ in 0..iterations {
                            enc.copy_buffer_to_buffer(
                                &self.buf_vel,
                                0,
                                &self.buf_vel_in,
                                0,
                                vel_bytes,
                            );
                            dispatch(&mut enc, &self.pipelines.collide, g);
                        }
                    }
                    Pass::X => dispatch(&mut enc, &self.pipelines.force_x, g),
                    Pass::Y => dispatch(&mut enc, &self.pipelines.force_y, g),
                    Pass::Radial => dispatch(&mut enc, &self.pipelines.radial, g),
                    Pass::Center => {
                        dispatch(&mut enc, &self.pipelines.center_reduce1, g);
                        dispatch(&mut enc, &self.pipelines.center_reduce2, 1);
                        dispatch(&mut enc, &self.pipelines.center_apply, g);
                    }
                }
            }
            dispatch(&mut enc, &self.pipelines.integrate, g);
            queue.submit(Some(enc.finish()));
        }
        self.positions_dirty = true;
    }

    /// Tick until `alpha < alpha_min` or `max_ticks`. Returns ticks run.
    pub fn run(&mut self, max_ticks: usize) -> usize {
        let mut t = 0;
        while self.params.alpha >= self.alpha_min && t < max_ticks {
            self.tick(1);
            t += 1;
        }
        t
    }

    fn read_back(&mut self) {
        let device = &self.gpu.device;
        let bytes = (8 * self.n) as u64;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        enc.copy_buffer_to_buffer(&self.buf_pos, 0, &self.buf_staging, 0, bytes);
        self.gpu.queue.submit(Some(enc.finish()));
        let slice = self.buf_staging.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device poll");
        rx.recv().expect("map result").expect("map positions");
        {
            let view = slice.get_mapped_range().expect("mapped range");
            self.positions.copy_from_slice(bytemuck::cast_slice(&view));
        }
        self.buf_staging.unmap();
        self.positions_dirty = false;
    }

    /// Interleaved `[x0, y0, x1, y1, …]`, read back from the GPU if stale.
    pub fn positions(&mut self) -> &[f32] {
        if self.positions_dirty {
            self.read_back();
        }
        &self.positions
    }

    #[allow(dead_code)]
    fn velocities_buffer(&self) -> &wgpu::Buffer {
        &self.buf_vel
    }
}
