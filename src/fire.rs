// base

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*base][base:1]]
//! The Fast-Inertial-Relaxation-Engine (FIRE) algorithm
#[derive(Debug, Clone)]
pub struct FIRE {
    /// the maximum time step allowed
    dt_max: f64,

    /// factor used to decrease alpha-parameter if downhill
    f_alpha: f64,

    /// initial alpha-parameter
    alpha_start: f64,

    /// the maximum displacement allowed
    maxdisp: f64,

    /// factor used to increase time-step if downhill
    f_inc: f64,

    /// factor used to decrease time-step if uphill
    f_dec: f64,

    /// minimum number of iterations ("latency" time) performed before time-step
    /// may be increased, which is important for the stability of the algorithm.
    nsteps_min: usize,

    /// adaptive time step for integration of the equations of motion
    dt: f64,

    /// adaptive parameter that controls the velocity used to evolve the system.
    alpha: f64,

    /// current velocity
    velocity: Option<Velocity>,

    /// displacement vector
    displacement: Option<Displacement>,

    /// current number of iterations when go downhill
    nsteps: usize,
}

impl Default for FIRE {
    fn default() -> Self {
        FIRE {
            // default parameters taken from the original paper
            dt_max      : 1.00,
            alpha_start : 0.10,
            f_alpha     : 0.99,
            f_dec       : 0.50,
            f_inc       : 1.10,
            maxdisp     : 0.10,
            nsteps_min  : 5,

            // counters or adaptive parameters
            dt           : 0.10,
            alpha        : 0.10,
            nsteps       : 0,
            velocity     : None,
            displacement : None,
        }
    }
}
// base:1 ends here

// core

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*core][core:1]]
impl FIRE {
    /// Propagate the system for one simulation step using FIRE algorithm.
    pub fn propagate(mut self, force: &[f64]) {
        // F0. prepare data
        let n = force.len();
        let mut velocity = self.velocity.unwrap_or(Velocity(vec![0.0; n]));
        let mut displacement = self.displacement.unwrap_or(Displacement(vec![0.0; n]));

        // F1. calculate the power: P = F·V
        let power = force.vecdot(&velocity);

        // F2. adjust velocity
        velocity.adjust(force, self.alpha);

        // F3 & F4: check the direction of power: go downhill or go uphill
        if power.is_sign_positive() {
            // F3. when go downhill
            // increase time step if we have go downhill for enough times
            if self.nsteps > self.nsteps_min {
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
        displacement.take_md_step(force, &velocity, self.dt);
        // scale the displacement according to max displacement
        displacement.rescale(self.maxdisp);

        // save state
        self.velocity = Some(velocity);
        self.displacement = Some(displacement);
    }
}
// core:1 ends here

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

    /// Update velocity
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
}
// velocity:1 ends here
