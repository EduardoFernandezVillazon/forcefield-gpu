//! Build a forcefield (CPU) simulation from a [`Config`].

use crate::spec::{Config, ForceSpec, NodeInit, Values};
use forcefield::forces::{Accessor, LinkRef};
use forcefield::forces::{Center, Collide, Link, ManyBody, Radial, X, Y};
use forcefield::{Node, Simulation};

fn node_acc(v: &Values) -> Accessor<usize> {
    match v {
        Values::Constant(c) => Accessor::Constant(*c as f64),
        Values::Each(e) => Accessor::Values(e.iter().map(|x| *x as f64).collect()),
    }
}

fn link_acc(v: &Values) -> Accessor<LinkRef> {
    match v {
        Values::Constant(c) => Accessor::Constant(*c as f64),
        Values::Each(e) => Accessor::Values(e.iter().map(|x| *x as f64).collect()),
    }
}

pub fn build(nodes: &[NodeInit], config: &Config) -> Simulation {
    let ns: Vec<Node> = nodes
        .iter()
        .map(|n| Node {
            x: n.x as f64,
            y: n.y as f64,
            vx: 0.0,
            vy: 0.0,
            fx: n.fx as f64,
            fy: n.fy as f64,
        })
        .collect();
    let mut sim = Simulation::new(&ns);
    sim.set_alpha(config.alpha as f64)
        .set_alpha_min(config.alpha_min as f64)
        .set_alpha_decay(config.alpha_decay as f64)
        .set_alpha_target(config.alpha_target as f64)
        .set_velocity_decay(config.velocity_decay as f64);
    for (name, spec) in &config.forces {
        let force: Box<dyn forcefield::Force> = match spec {
            ForceSpec::ManyBody {
                strength,
                theta,
                distance_min,
                distance_max,
            } => Box::new(
                ManyBody::new()
                    .strength(node_acc(strength))
                    .theta(*theta as f64)
                    .distance_min(*distance_min as f64)
                    .distance_max(*distance_max as f64),
            ),
            ForceSpec::Link {
                links,
                distance,
                strength,
                iterations,
            } => {
                let mut f = Link::new(
                    links
                        .iter()
                        .map(|&(s, t)| (s as usize, t as usize))
                        .collect(),
                )
                .distance(link_acc(distance))
                .iterations(*iterations as usize);
                if let Some(s) = strength {
                    f = f.strength(link_acc(s));
                }
                Box::new(f)
            }
            ForceSpec::Collide {
                radius,
                strength,
                iterations,
            } => Box::new(
                Collide::new(node_acc(radius))
                    .strength(*strength as f64)
                    .iterations(*iterations as usize),
            ),
            ForceSpec::X { x, strength } => {
                Box::new(X::new(node_acc(x)).strength(node_acc(strength)))
            }
            ForceSpec::Y { y, strength } => {
                Box::new(Y::new(node_acc(y)).strength(node_acc(strength)))
            }
            ForceSpec::Center { x, y, strength } => {
                Box::new(Center::new(*x as f64, *y as f64).strength(*strength as f64))
            }
            ForceSpec::Radial {
                radius,
                x,
                y,
                strength,
            } => Box::new(
                Radial::new(node_acc(radius), *x as f64, *y as f64).strength(node_acc(strength)),
            ),
        };
        sim.add_force(name.clone(), force);
    }
    sim
}
