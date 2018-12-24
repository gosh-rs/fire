// base

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*base][base:1]]
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

    /// current velocities
    velocities: Option<Velocities>,

    /// current number of iterations when go downhill
    nsteps: usize,
}

impl Default for FIRE {
    fn default() -> Self {
        FIRE {
            // default parameters taken from the original paper
            dt_max     : 1.00,
            alpha_start: 0.10,
            f_alpha    : 0.99,
            f_dec      : 0.50,
            f_inc      : 1.10,
            maxdisp    : 0.10,
            nsteps_min : 5,

            // counters or adaptive parameters
            dt         : 0.10,
            alpha      : 0.10,
            nsteps     : 0,
            velocities : None,
        }
    }
}
// base:1 ends here

// core

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*core][core:1]]
impl FIRE {
    /// Propagate the system for one simulation step using FIRE algorithm.
    fn propagate(mut self, forces: &[f64]) -> Result<Displacement, String> {
        // F0. prepare data
        let n = forces.len();
        let mut velocities = self.velocities.unwrap_or(Velocities(vec![0.0; n]));

        // F1. calculate the power: P = F·V
        let power = forces.vecdot(&velocities);

        // F2. adjust velocities
        velocities.adjust(forces, self.alpha);

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
            // reset velocities to zero
            velocities.reset();
        }

        // F5. calculate displacement vectors based on a typical MD stepping algorithm
        // update the internal velocities
        let mut disp = Displacement(vec![0.0; n]);
        disp.take_md_step(forces, &velocities, self.dt);

        // scale the displacement according to max displacement
        disp.rescale(self.maxdisp);

        // save state
        self.velocities = Some(velocities);

        Ok(disp)
    }
}
// core:1 ends here

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*Displacement][Displacement:3]]
use crate::math::*;

#[derive(Debug, Clone)]
pub struct Displacement(Vec<f64>);

impl Displacement {
    /// Get particle displacement vectors by performing a regular MD step
    ///
    /// D = dt * V + 0.5 * F * dt^2
    pub fn take_md_step(&mut self, forces: &[f64], velocities: &[f64], timestep: f64) {
        let n = velocities.len();
        debug_assert!(n == forces.len(), "input vectors are in different size!");

        let dt = timestep;

        // Verlet algorithm
        self.0 = velocities.to_vec();

        // D = dt * V
        self.0.vecscale(dt);

        // D += 0.5 * dt^2 * F
        self.0.vecadd(forces, 0.5 * dt.powi(2));
    }

    // scale the displacement vectors if exceed a given max displacement.
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
// Displacement:3 ends here

// Velocities

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*Velocities][Velocities:1]]
#[derive(Debug, Clone)]
pub struct Velocities(Vec<f64>);

impl Deref for Velocities {
    type Target = Vec<f64>;

    fn deref(&self) -> &Vec<f64> {
        &self.0
    }
}

impl Velocities {
    // reset velocities to zero
    pub fn reset(&mut self) {
        for i in 0..self.0.len() {
            self.0[i] = 0.0;
        }
    }

    /// Update velocities
    ///
    /// V = (1 - alpha) · V + alpha · F / |F| · |V|
    pub fn adjust(&mut self, forces: &[f64], alpha: f64) {
        let vnorm = self.0.vec2norm();
        let fnorm = forces.vec2norm();

        // V = (1-alpha) · V
        self.0.vecscale(1.0 - alpha);

        // V += alpha · F / |F| · |V|
        self.0.vecadd(forces, alpha * vnorm / fnorm);
    }
}
// Velocities:1 ends here
