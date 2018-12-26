// header

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*header][header:1]]
//! The Fast-Inertial-Relaxation-Engine (FIRE) algorithm
///
/// This method is stable with respect to random errors in the potential energy.
// header:1 ends here

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
    pub max_step: f64,

    /// Factor used to decrease alpha-parameter if downhill
    pub f_alpha: f64,

    /// Initial alpha-parameter.
    pub alpha_start: f64,

    /// Factor used to increase time-step if downhill
    pub f_inc: f64,

    /// Factor used to decrease time-step if uphill
    pub f_dec: f64,

    /// Minimum number of iterations ("latency" time) performed before time-step
    /// may be increased, which is important for the stability of the algorithm.
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
    pub displacement: Option<Displacement>,

    /// Current number of iterations when go downhill
    nsteps: usize,
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
        }
    }
}
// base:2 ends here

// builder

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*builder][builder:1]]
impl FIRE {
    /// Sets the maximum size for an optimization step.
    pub fn with_max_step(mut self, maxstep: f64) -> Self {
        self.max_step = maxstep;
        self
    }
}
// builder:1 ends here

// core

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*core][core:1]]
impl FIRE {
    /// Propagate the system for one simulation step using FIRE algorithm.
    fn propagate(mut self, force: &[f64]) -> Self {
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
        velocity.update(force, self.dt);

        // let mut displacement = Displacement(vec![0.0; n]);
        displacement.take_md_step(force, &velocity, self.dt);

        // scale the displacement according to max displacement
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
impl FIRE {
    pub fn minimize<E, S>(mut self, x: &mut [f64], mut f: E, mut stopping: S)
    where
        E: FnMut(&[f64], &mut [f64]) -> f64,
        S: TerminationCriteria,
    {
        let n = x.len();
        let mut forces = vec![0.0; n];

        for i in 0.. {
            let fx = f(x, &mut forces);

            let progress = Progress {
                niter: i,
                gnorm: forces.vec2norm(),
                fx,
            };

            if stopping.met(&progress) {
                println!("normal termination!");
                break;
            }

            self = self.propagate(&forces);
            if let Some(ref displ) = self.displacement {
                displ.apply(x);
            } else {
                panic!("bad");
            }
        }
    }
}
// entry:1 ends here

// stop

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*stop][stop:1]]
/// Termination criteria
#[derive(Debug, Clone)]
pub struct Termination {
    /// The maximum number of optimization cycles.
    pub max_cycles: usize,
    /// The max allowed gradient norm
    pub max_gradient_norm: f64,
}

impl Default for Termination {
    fn default() -> Self {
        Termination {
            max_cycles: 0,
            max_gradient_norm: 0.2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Progress {
    pub fx: f64,
    pub niter: usize,
    pub gnorm: f64,
}

pub trait TerminationCriteria {
    fn met(&mut self, progress: &Progress) -> bool;
}

impl TerminationCriteria for Termination {
    fn met(&mut self, progress: &Progress) -> bool {
        if self.max_cycles > 0 && progress.niter >= self.max_cycles {
            return true;
        }

        if progress.gnorm <= self.max_gradient_norm {
            return true;
        }

        false
    }
}

/// For user defined termination criteria
pub struct UserTermination<G>
where
    G: FnMut(&Progress) -> bool,
{
    cb: G,
}

impl<G> UserTermination<G>
where
    G: FnMut(&Progress) -> bool,
{
    pub fn new(cb: G) -> Self {
        UserTermination { cb }
    }
}

impl<G> TerminationCriteria for UserTermination<G>
where
    G: FnMut(&Progress) -> bool,
{
    fn met(&mut self, progress: &Progress) -> bool {
        (self.cb)(progress)
    }
}
// stop:1 ends here

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*displacement][displacement:3]]
use vecfx::*;

/// Represents the displacement vector
#[derive(Debug, Clone)]
pub struct Displacement(Vec<f64>);

impl Displacement {
    /// Get particle displacement vectors by performing a regular MD step
    ///
    /// D = dt * V + 0.5 * F * dt^2
    pub fn take_md_step(&mut self, force: &[f64], velocity: &[f64], timestep: f64) {
        let n = velocity.len();
        debug_assert!(
            n == force.len(),
            "the sizes of input vectors are different!"
        );

        let dt = timestep;

        // Verlet algorithm
        self.0 = velocity.to_vec();

        // D = dt * V
        self.0.vecscale(dt);

        // D += 0.5 * dt^2 * F
        self.0.vecadd(force, 0.5 * dt.powi(2));
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

    /// adjust velocity
    ///
    /// V = (1 - alpha) · V + alpha · F / |F| · |V|
    pub fn adjust(&mut self, force: &[f64], alpha: f64) {
        let vnorm = self.0.vec2norm();
        let fnorm = force.vec2norm();

        // V = (1-alpha) · V
        self.0.vecscale(1.0 - alpha);

        // V += alpha · F / |F| · |V|
        self.0.vecadd(force, alpha * vnorm / fnorm);
    }

    /// update velocity
    ///
    /// V += dt · F
    pub fn update(&mut self, force: &[f64], dt: f64) {
        self.0.vecadd(force, dt);
    }
}
// velocity:1 ends here
