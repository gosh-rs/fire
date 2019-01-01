// header

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*header][header:1]]
//! The Fast-Inertial-Relaxation-Engine (FIRE) algorithm
///
/// This method is stable with respect to random errors in the potential energy
/// and force. FIRE has strict adherence to minimizing forces, which makes
/// constrained minimization easy.

use crate::Progress;
use crate::Termination;
use crate::UserTermination;
use crate::TerminationCriteria;
use crate::GradientBasedMinimizer;
// header:1 ends here

// scheme

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*scheme][scheme:1]]
/// MD Integration formulations for position update and velocity update
#[derive(Clone, Copy, Debug)]
pub enum MdScheme {
    ForwardEuler,
    VelocityVerlet,
    SemiImplicitEuler,
}

impl Default for MdScheme {
    fn default() -> Self {
        MdScheme::VelocityVerlet
    }
}
// scheme:1 ends here

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*base][base:2]]
/// The Fast-Inertial-Relaxation-Engine (FIRE) algorithm
///
/// # Notes from the paper
/// 
/// * `n_min` larger than 1 (at least a few smooth steps after freezing);
/// * `f_inc` larger than but near to one (avoid too fast acceleration);
/// * `f_dec` smaller than 1 but much larger than zero (avoid too heavy slowing down)
/// * `alpha_start` larger than, but near to zero (avoid too heavy damping)
/// * `f_alpha` smaller than, but near to one (mixing is efficient some time after restart).
#[derive(Debug, Clone)]
pub struct FIRE {
    /// The maximum size for an optimization step. According to the paper, this
    /// is the only parameter needs to be adjusted by the user.
    ///
    /// The default value is 0.10.
    pub max_step: f64,

    /// Factor used to decrease alpha-parameter if downhill
    ///
    /// The default value is 0.99.
    pub f_alpha: f64,

    /// Initial alpha-parameter.
    ///
    /// The default value is 0.1.
    pub alpha_start: f64,

    /// Factor used to increase time-step if downhill
    ///
    /// The default value is 1.1.
    pub f_inc: f64,

    /// Factor used to decrease time-step if uphill
    ///
    /// The default value is 0.5.
    pub f_dec: f64,

    /// Minimum number of iterations ("latency" time) performed before time-step
    /// may be increased, which is important for the stability of the algorithm.
    ///
    /// The default value is 5.
    pub n_min: usize,

    /// adaptive time step for integration of the equations of motion
    dt: f64,

    /// The maximum time step as in the paper. Do not change this, change
    /// max_step instead.
    dt_max: f64,

    /// Adaptive parameter that controls the velocity used to evolve the system.
    alpha: f64,

    /// Current velocity
    velocity: Option<Velocity>,

    /// Displacement vector
    displacement: Option<Displacement>,

    /// Current number of iterations when go downhill
    nsteps: usize,

    /// Default termination criteria
    termination: Termination,

    /// MD scheme
    scheme: MdScheme,
}

impl Default for FIRE {
    fn default() -> Self {
        FIRE {
            // default parameters taken from the original paper
            dt_max: 1.00,
            alpha_start: 0.10,
            f_alpha: 0.99,
            f_dec: 0.50,
            f_inc: 1.10,
            // pele: 0.5, ase: 0.2
            max_step: 0.10,
            n_min: 5,

            // counters or adaptive parameters
            dt: 0.10,
            alpha: 0.10,
            nsteps: 0,
            velocity: None,
            displacement: None,

            // others
            termination: Termination::default(),
            scheme: MdScheme::default(),
        }
    }
}
// base:2 ends here

// builder

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*builder][builder:1]]
impl FIRE {
    /// Set the maximum size for an optimization step.
    pub fn with_max_step(mut self, maxstep: f64) -> Self {
        assert!(
            maxstep.is_sign_positive(),
            "step size should be a positive number!"
        );

        self.max_step = maxstep;
        self
    }

    /// Set MD scheme for position and velocity update
    pub fn with_md_scheme(mut self, scheme: &str) -> Self {
        match scheme {
            "SE" => self.scheme = MdScheme::SemiImplicitEuler,
            "VV" => self.scheme = MdScheme::VelocityVerlet,
            "FE" => self.scheme = MdScheme::ForwardEuler,
            _ => unimplemented!(),
        }
        self
    }

    /// Set the maximum cycles for termination.
    pub fn with_max_cycles(mut self, n: usize) -> Self {
        self.termination.max_cycles = n;
        self
    }

    /// Set the maximum gradient/force norm for termination.
    pub fn with_max_gradient_norm(mut self, gmax: f64) -> Self {
        assert!(gmax.is_sign_positive(), "gmax: bad parameter!");
        self.termination.max_gradient_norm = gmax;

        self
    }
}
// builder:1 ends here

// core

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*core][core:1]]
impl FIRE {
    /// Propagate the system for one simulation step using FIRE algorithm.
    fn propagate(mut self, force_prev: &[f64], force: &[f64]) -> Self {
        // F0. prepare data
        let n = force.len();
        let mut velocity = self.velocity.unwrap_or(Velocity(vec![0.0; n]));
        // caching displacement for memory performance
        let mut displacement = self.displacement.unwrap_or(Displacement(vec![0.0; n]));

        // F1. calculate the power: P = F·V
        let power = force.vecdot(&velocity);

        // F2. adjust velocity
        velocity.adjust(force, self.alpha);

        // F3 & F4: check the direction of power: go downhill or go uphill
        if power.is_sign_positive() {
            // F3. when go downhill
            // increase time step if we have go downhill for enough times
            if self.nsteps > self.n_min {
                self.dt = self.dt_max.min(self.dt * self.f_inc);
                self.alpha *= self.f_alpha;
            }
            // increment step counter
            self.nsteps += 1;
        } else {
            // F4. when go uphill
            // decrease time-step
            self.dt *= self.f_dec;
            // reset alpha
            self.alpha = self.alpha_start;
            // reset step counter
            self.nsteps = 0;
            // reset velocity to zero
            velocity.reset();
        }

        // F5. calculate displacement vectors based on a typical MD stepping algorithm
        // update the internal velocity
        displacement.take_md_step(&force_prev, &velocity, self.dt, self.scheme);
        velocity.take_md_step(&force_prev, &force, self.dt, self.scheme);

        // scale the displacement according to max step
        displacement.rescale(self.max_step);

        // save state
        self.velocity = Some(velocity);
        self.displacement = Some(displacement);
        self
    }
}
// core:1 ends here

// entry

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*entry][entry:1]]
use crate::line::*;

impl GradientBasedMinimizer for FIRE {
    /// minimize with user defined termination criteria / monitor
    fn minimize_alt<E, G>(mut self, x: &mut [f64], mut f: E, mut stopping: Option<G>)
    where
        E: FnMut(&[f64], &mut [f64]) -> f64,
        G: TerminationCriteria,
    {
        let n = x.len();
        let mut x_prev = x.to_vec();

        let mut gx = vec![0.0; n];
        let mut gx_prev = gx.clone();

        // evaluate first
        let mut fx = f(x, &mut gx);
        let mut fx_prev = fx;

        let ls = linesearch()
            .with_max_iterations(20)
            .with_algorithm("BackTracking");
            // .with_algorithm("MoreThuente");

        let mut ncall = 1;
        for i in 1.. {
            // cache gradient of previous step
            // to force
            gx.vecscale(-1.0);
            self = self.propagate(&gx_prev, &gx);

            // save previous point
            gx_prev.veccpy(&gx);
            x_prev.veccpy(&x);
            fx_prev = fx;

            // determine optimal step size along displacement
            if let Some(ref d) = self.displacement {
                // to gradient
                gx.vecscale(-1.0);
                // perform line search
                let mut dg = 0.0;
                let mut step = 1.0;
                let phi = |stp: f64| {
                    // current point or trial point
                    if stp == 0.0 {
                        (fx, d.vecdot(&gx))
                    } else {
                        // restore position
                        x.veccpy(&x_prev);
                        x.vecadd(&d, stp);
                        // update value and gradient
                        let r = f(x, &mut gx);
                        // update data
                        ncall += 1;
                        fx = r;
                        step = stp;

                        (r, d.vecdot(&gx))
                    }
                };
                let _ = ls.find(phi).expect("ls");

                let progress = Progress {
                    niter: i,
                    gx: &gx,
                    gnorm: gx.vec2norm(),
                    displacement: d,
                    x,
                    fx,
                    step,
                    ncall,
                };

                if let Some(ref mut stopping) = stopping {
                    if stopping.met(&progress) {
                        break;
                    }
                }

                if self.termination.met(&progress) {
                    println!("normal termination!");
                    break;
                }
            } else {
                panic!("bad");
            }
        }
    }
}
// entry:1 ends here

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*displacement][displacement:3]]
use vecfx::*;

/// Represents the displacement vector
#[derive(Debug, Clone)]
pub struct Displacement(Vec<f64>);

impl Displacement {
    /// Update particle displacement vector by performing a regular MD step
    ///
    /// Displacement = X(n+1) - X(n)
    ///
    pub fn take_md_step(
        &mut self,
        force: &[f64],    // F(n)
        velocity: &[f64], // V(n)
        timestep: f64,    // Δt
        scheme: MdScheme,
    ) {
        debug_assert!(
            velocity.len() == force.len()
            "the sizes of input vectors are different!"
        );

        let dt = timestep;
        self.0 = velocity.to_vec();
        match scheme {
            // Velocity Verlet (VV) integration formula.
            //
            // X(n+1) - X(n) = dt * V(n) + 0.5 * F(n) * dt^2
            MdScheme::VelocityVerlet => {
                // D = dt * V
                self.0.vecscale(dt);
                // D += 0.5 * dt^2 * F
                self.0.vecadd(force, 0.5 * dt.powi(2));
            }
            // Semi-implicit Euler (SE) or Forward Euler (FE)
            //
            // D = dt * V
            MdScheme::SemiImplicitEuler | MdScheme::ForwardEuler => {
                self.0.vecscale(timestep);
            }
        }
    }

    /// Scale the displacement vector if its norm exceeds a given cutoff.
    pub fn rescale(&mut self, max_disp: f64) {
        // get the max norm of displacement vector for atoms
        let norm = self.0.vec2norm();

        // scale the displacement vectors if too large
        if norm > max_disp {
            let s = max_disp / norm;
            self.0.vecscale(s);
        }
    }

    /// Apply displacement vector to `x`
    /// x += d
    pub fn apply(&self, x: &mut [f64]) {
        x.vecadd(self, 1.0);
    }
}

// Deref coercion: use Displacement as a normal vec
use std::ops::Deref;
impl Deref for Displacement {
    type Target = Vec<f64>;

    fn deref(&self) -> &Vec<f64> {
        &self.0
    }
}
// displacement:3 ends here

// velocity

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*velocity][velocity:1]]
/// Represents the velocity vector
#[derive(Debug, Clone)]
pub struct Velocity(Vec<f64>);

impl Deref for Velocity {
    type Target = Vec<f64>;

    fn deref(&self) -> &Vec<f64> {
        &self.0
    }
}

impl Velocity {
    /// Reset velocity to zero
    pub fn reset(&mut self) {
        for i in 0..self.0.len() {
            self.0[i] = 0.0;
        }
    }

    /// Adjust velocity
    /// V = (1 - alpha) · V + alpha · F / |F| · |V|
    ///
    pub fn adjust(&mut self, force: &[f64], alpha: f64) {
        let vnorm = self.0.vec2norm();
        let fnorm = force.vec2norm();

        // V = (1-alpha) · V
        self.0.vecscale(1.0 - alpha);

        // V += alpha · F / |F| · |V|
        self.0.vecadd(force, alpha * vnorm / fnorm);
    }

    /// Update velocity using Velocity Verlet (VV) formulation.
    /// V(n+1) += dt· F(n)
    pub fn take_md_step(
        &mut self,
        force_this: &[f64], // F(n)
        force_next: &[f64], // F(n+1)
        dt: f64,            // Δt
        scheme: MdScheme,
    ) {
        match scheme {
            // Update velocity using Velocity Verlet (VV) formulation.
            //
            // V(n+1) = V(n) + 0.5 · dt · [F(n) + F(n+1)]
            MdScheme::VelocityVerlet => {
                self.0.vecadd(force_this, 0.5 * dt);
                self.0.vecadd(force_next, 0.5 * dt);
            }
            // V(n+1) = V(n) + dt· F(n+1)
            MdScheme::SemiImplicitEuler => {
                self.0.vecadd(force_next, dt);
            }
            // V(n+1) = V(n) + dt· F(n)
            MdScheme::ForwardEuler => {
                self.0.vecadd(force_this, dt);
            }
        }
    }
}
// velocity:1 ends here
