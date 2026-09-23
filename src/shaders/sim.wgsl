// The d3-force model as compute passes. One pass per force, dispatched in
// the configured order; `integrate` last. Forces that d3 applies pairwise
// with sequential updates (link, collide) are computed as per-node gathers
// from a velocity snapshot (`vel_in`), i.e. Jacobi instead of Gauss-Seidel.

struct Params {
  n: u32,
  m: u32,
  alpha: f32,
  velocity_decay: f32,        // stored as d3 does: 1 - user value
  distance_min2: f32,
  distance_max2: f32,
  collide_strength: f32,
  center_strength: f32,
  center_x: f32,
  center_y: f32,
  radial_x: f32,
  radial_y: f32,
  tick: u32,
  groups: u32,                // workgroups over n (for the reduction)
  ptr_off: u32,               // offsets into links_u
  idx_off: u32,
  ends_off: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

// per-node floats, stride NF: strength, radius, xz, xs, yz, ys, rr, rs, fx, fy, fmx, fmy
const NF: u32 = 12u;
// per-link floats, stride LF: distance, strength, bias
const LF: u32 = 3u;
const WG: u32 = 256u;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> vel_in: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> nodes_f: array<f32>;
@group(0) @binding(5) var<storage, read> links_u: array<u32>;
@group(0) @binding(6) var<storage, read> links_f: array<f32>;
@group(0) @binding(7) var<storage, read_write> reduce_buf: array<vec2<f32>>;

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn jiggle(seed: u32) -> f32 {
  return (f32(pcg(seed)) / 4294967296.0 - 0.5) * 1e-6;
}

var<workgroup> tile_pos: array<vec2<f32>, 256>;
var<workgroup> tile_s: array<f32, 256>;

// ---- many-body: every pair, tiled through workgroup memory --------------
@compute @workgroup_size(256)
fn many_body(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = gid.x;
  let n = P.n;
  var p = vec2<f32>(0.0);
  if (i < n) { p = pos[i]; }
  var dv = vec2<f32>(0.0);
  let tiles = (n + WG - 1u) / WG;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let j = t * WG + lid.x;
    if (j < n) { tile_pos[lid.x] = pos[j]; tile_s[lid.x] = nodes_f[j * NF]; }
    else { tile_pos[lid.x] = vec2<f32>(0.0); tile_s[lid.x] = 0.0; }
    workgroupBarrier();
    if (i < n) {
      for (var k = 0u; k < WG; k = k + 1u) {
        let j2 = t * WG + k;
        if (j2 >= n || j2 == i) { continue; }
        var d = tile_pos[k] - p;
        var l = dot(d, d);
        if (l >= P.distance_max2) { continue; }
        if (d.x == 0.0) { d.x = jiggle(i * 2654435761u + j2 * 40503u + P.tick); l = l + d.x * d.x; }
        if (d.y == 0.0) { d.y = jiggle(i * 2654435761u + j2 * 40503u + P.tick + 7919u); l = l + d.y * d.y; }
        if (l < P.distance_min2) { l = sqrt(P.distance_min2 * l); }
        let w = tile_s[k] * P.alpha / l;
        dv = dv + d * w;
      }
    }
    workgroupBarrier();
  }
  if (i < n) { vel[i] = vel[i] + dv; }
}

// ---- link: gather over incident links from the velocity snapshot ---------
@compute @workgroup_size(256)
fn link(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let start = links_u[P.ptr_off + i];
  let end = links_u[P.ptr_off + i + 1u];
  var dv = vec2<f32>(0.0);
  for (var k = start; k < end; k = k + 1u) {
    let e = links_u[P.idx_off + k];
    let s = links_u[P.ends_off + 2u * e];
    let t = links_u[P.ends_off + 2u * e + 1u];
    var d = pos[t] + vel_in[t] - pos[s] - vel_in[s];
    if (d.x == 0.0) { d.x = jiggle(e * 2654435761u + P.tick); }
    if (d.y == 0.0) { d.y = jiggle(e * 2654435761u + P.tick + 7919u); }
    var l = length(d);
    l = (l - links_f[e * LF]) / l * P.alpha * links_f[e * LF + 1u];
    d = d * l;
    let b = links_f[e * LF + 2u];
    if (i == t) { dv = dv - d * b; }
    if (i == s) { dv = dv + d * (1.0 - b); }
  }
  vel[i] = vel[i] + dv;
}

// ---- collide: every pair, on predicted positions from the snapshot -------
@compute @workgroup_size(256)
fn collide(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = gid.x;
  let n = P.n;
  var xi = vec2<f32>(0.0);
  var ri = 0.0;
  if (i < n) { xi = pos[i] + vel_in[i]; ri = nodes_f[i * NF + 1u]; }
  let ri2 = ri * ri;
  var dv = vec2<f32>(0.0);
  let tiles = (n + WG - 1u) / WG;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let j = t * WG + lid.x;
    if (j < n) { tile_pos[lid.x] = pos[j] + vel_in[j]; tile_s[lid.x] = nodes_f[j * NF + 1u]; }
    else { tile_pos[lid.x] = vec2<f32>(1e30); tile_s[lid.x] = 0.0; }
    workgroupBarrier();
    if (i < n) {
      for (var k = 0u; k < WG; k = k + 1u) {
        let j2 = t * WG + k;
        if (j2 >= n || j2 == i) { continue; }
        let rj = tile_s[k];
        let r = ri + rj;
        var d = xi - tile_pos[k];
        var l = dot(d, d);
        if (l >= r * r) { continue; }
        if (d.x == 0.0) { d.x = jiggle(i * 2654435761u + j2 * 40503u + P.tick + 13u); l = l + d.x * d.x; }
        if (d.y == 0.0) { d.y = jiggle(i * 2654435761u + j2 * 40503u + P.tick + 17u); l = l + d.y * d.y; }
        l = sqrt(l);
        l = (r - l) / l * P.collide_strength;
        d = d * l;
        let rj2 = rj * rj;
        dv = dv + d * (rj2 / (ri2 + rj2));
      }
    }
    workgroupBarrier();
  }
  if (i < n) { vel[i] = vel[i] + dv; }
}

// ---- x / y / radial ---------------------------------------------------------
@compute @workgroup_size(256)
fn force_x(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  vel[i].x = vel[i].x + (nodes_f[i * NF + 2u] - pos[i].x) * nodes_f[i * NF + 3u] * P.alpha;
}
@compute @workgroup_size(256)
fn force_y(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  vel[i].y = vel[i].y + (nodes_f[i * NF + 4u] - pos[i].y) * nodes_f[i * NF + 5u] * P.alpha;
}
@compute @workgroup_size(256)
fn radial(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  var d = pos[i] - vec2<f32>(P.radial_x, P.radial_y);
  if (d.x == 0.0) { d.x = 1e-6; }
  if (d.y == 0.0) { d.y = 1e-6; }
  let r = length(d);
  let k = (nodes_f[i * NF + 6u] - r) * nodes_f[i * NF + 7u] * P.alpha / r;
  vel[i] = vel[i] + d * k;
}

// ---- center: two-level reduction of the mean position, then translate ----
var<workgroup> red: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn center_reduce1(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let i = gid.x;
  red[lid.x] = select(vec2<f32>(0.0), pos[i], i < P.n);
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { reduce_buf[wid.x] = red[0]; }
}
@compute @workgroup_size(256)
fn center_reduce2(@builtin(local_invocation_id) lid: vec3<u32>) {
  var acc = vec2<f32>(0.0);
  for (var g = lid.x; g < P.groups; g = g + WG) { acc = acc + reduce_buf[g]; }
  red[lid.x] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let mean = red[0] / f32(P.n);
    reduce_buf[0] = (mean - vec2<f32>(P.center_x, P.center_y)) * P.center_strength;
  }
}
@compute @workgroup_size(256)
fn center_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  pos[i] = pos[i] - reduce_buf[0];
}

// ---- integrate ------------------------------------------------------------
@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let base = i * NF;
  let fm = vec2<f32>(nodes_f[base + 10u], nodes_f[base + 11u]);
  let f = vec2<f32>(nodes_f[base + 8u], nodes_f[base + 9u]);
  var v = vel[i] * P.velocity_decay;
  var p = pos[i] + v;
  // fixed axes: position pinned, velocity zeroed
  p = mix(p, f, fm);
  v = mix(v, vec2<f32>(0.0), fm);
  vel[i] = v;
  pos[i] = p;
}
