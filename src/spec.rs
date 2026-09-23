//! A declarative description of a simulation: the same model as forcefield
//! (d3-force), expressed as data so one spec can build either backend.

/// A per-node or per-link parameter: one value for all, or one per item.
#[derive(Clone, Debug, PartialEq)]
pub enum Values {
    Constant(f32),
    Each(Vec<f32>),
}

impl Values {
    pub fn get(&self, i: usize) -> f32 {
        match self {
            Values::Constant(v) => *v,
            Values::Each(v) => v[i],
        }
    }
    pub fn to_vec(&self, n: usize) -> Vec<f32> {
        match self {
            Values::Constant(v) => vec![*v; n],
            Values::Each(v) => {
                assert_eq!(
                    v.len(),
                    n,
                    "Values::Each length {} != item count {n}",
                    v.len()
                );
                v.clone()
            }
        }
    }
}

impl From<f32> for Values {
    fn from(v: f32) -> Self {
        Values::Constant(v)
    }
}
impl From<Vec<f32>> for Values {
    fn from(v: Vec<f32>) -> Self {
        Values::Each(v)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ForceSpec {
    /// d3 `forceManyBody`. `theta` is honoured by the CPU backend only; the GPU
    /// backend computes every pair (theta 0).
    ManyBody {
        strength: Values,
        theta: f32,
        distance_min: f32,
        distance_max: f32,
    },
    /// d3 `forceLink`. `strength: None` selects d3's default `1 / min(degree)`.
    Link {
        links: Vec<(u32, u32)>,
        distance: Values,
        strength: Option<Values>,
        iterations: u32,
    },
    /// d3 `forceCollide`.
    Collide {
        radius: Values,
        strength: f32,
        iterations: u32,
    },
    /// d3 `forceX` / `forceY`.
    X {
        x: Values,
        strength: Values,
    },
    Y {
        y: Values,
        strength: Values,
    },
    /// d3 `forceCenter`: translates positions so their mean sits at (x, y).
    Center {
        x: f32,
        y: f32,
        strength: f32,
    },
    /// d3 `forceRadial`.
    Radial {
        radius: Values,
        x: f32,
        y: f32,
        strength: Values,
    },
}

impl ForceSpec {
    pub fn many_body(strength: impl Into<Values>) -> Self {
        ForceSpec::ManyBody {
            strength: strength.into(),
            theta: 0.9,
            distance_min: 1.0,
            distance_max: f32::INFINITY,
        }
    }
    pub fn link(links: Vec<(u32, u32)>) -> Self {
        ForceSpec::Link {
            links,
            distance: Values::Constant(30.0),
            strength: None,
            iterations: 1,
        }
    }
    pub fn collide(radius: impl Into<Values>) -> Self {
        ForceSpec::Collide {
            radius: radius.into(),
            strength: 1.0,
            iterations: 1,
        }
    }
    pub fn x(x: impl Into<Values>) -> Self {
        ForceSpec::X {
            x: x.into(),
            strength: Values::Constant(0.1),
        }
    }
    pub fn y(y: impl Into<Values>) -> Self {
        ForceSpec::Y {
            y: y.into(),
            strength: Values::Constant(0.1),
        }
    }
    pub fn center(x: f32, y: f32) -> Self {
        ForceSpec::Center {
            x,
            y,
            strength: 1.0,
        }
    }
}

/// Simulation parameters, with d3's defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub alpha: f32,
    pub alpha_min: f32,
    pub alpha_decay: f32,
    pub alpha_target: f32,
    /// User-facing value (d3 default 0.4).
    pub velocity_decay: f32,
    /// Applied in order every tick, as d3 does.
    pub forces: Vec<(String, ForceSpec)>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            alpha: 1.0,
            alpha_min: 0.001,
            alpha_decay: 1.0 - 0.001f32.powf(1.0 / 300.0),
            alpha_target: 0.0,
            velocity_decay: 0.4,
            forces: Vec::new(),
        }
    }
}

impl Config {
    pub fn force(mut self, name: impl Into<String>, spec: ForceSpec) -> Self {
        self.forces.push((name.into(), spec));
        self
    }
}

/// Initial node state; `NaN` = unset (spiral initialisation, free axis).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeInit {
    pub x: f32,
    pub y: f32,
    pub fx: f32,
    pub fy: f32,
}

impl NodeInit {
    pub const UNSET: NodeInit = NodeInit {
        x: f32::NAN,
        y: f32::NAN,
        fx: f32::NAN,
        fy: f32::NAN,
    };
    pub fn at(x: f32, y: f32) -> Self {
        NodeInit {
            x,
            y,
            ..NodeInit::UNSET
        }
    }
}

impl Default for NodeInit {
    fn default() -> Self {
        NodeInit::UNSET
    }
}

/// d3's phyllotaxis initial placement for node `i`.
pub fn spiral(i: usize) -> (f32, f32) {
    let radius = 10.0 * (0.5 + i as f64).sqrt();
    let angle = i as f64 * std::f64::consts::PI * (3.0 - 5f64.sqrt());
    ((radius * angle.cos()) as f32, (radius * angle.sin()) as f32)
}
