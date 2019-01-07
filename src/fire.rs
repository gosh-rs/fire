// header

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*header][header:1]]
//! The Fast-Inertial-Relaxation-Engine (FIRE) algorithm
///
/// This method is stable with respect to random errors in the potential energy
/// and force. FIRE has strict adherence to minimizing forces, which makes
/// constrained minimization easy.

use crate::Progress;
use crate::Termination;
use crate::TerminationCriteria;
use crate::GradientBasedMinimizer;
use crate::CachedProblem;
use crate::common::*;
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

    /// adaptive time step for integration of the equations of motion. The
    /// initial value is 0.1.
    dt: f64,

    /// The maximum time step as in the paper. Do not change this, change
    /// max_step instead.
    dt_max: f64,

    /// The minimum time step when decreasing `dt`. The default is 0.01.
    dt_min: f64,

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

    /// Apply line search for optimal step size.
    use_line_search: bool,
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
            dt_min: 0.02,
            alpha: 0.10,
            nsteps: 0,
            velocity: None,
            displacement: None,

            // others
            termination: Termination::default(),
            scheme: MdScheme::default(),
            use_line_search: false,
        }
    }
}
// base:2 ends here

// scheme

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*scheme][scheme:1]]
/// MD Integration formulations for position update and velocity update
#[derive(Clone, Copy, Debug)]
pub enum MdScheme {
    /// Velocity Verlet
    VelocityVerlet,
    /// Semi-implicit Euler algorithm. I think this is the algorithm implemented
    /// in ASE.
    SemiImplicitEuler,
}

impl Default for MdScheme {
    fn default() -> Self {
        MdScheme::VelocityVerlet
        // MdScheme::SemiImplicitEuler
    }
}
// scheme:1 ends here

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
            "ASE" => self.scheme = MdScheme::SemiImplicitEuler,
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

    /// Enable line search of optimal step size.
    ///
    /// The default is no line search.
    pub fn with_line_search(mut self) -> Self {
        self.use_line_search = true;
        self
    }
}
// builder:1 ends here

// core

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*core][core:1]]
impl FIRE {
    /// Propagate the system for one simulation step using FIRE algorithm.
    fn propagate<E>(
        mut self,
        force_prev: &mut [f64],
        force: &mut [f64],
        problem: &mut CachedProblem<E>,
    ) -> Self
    where
        E: FnMut(&[f64], &mut [f64]) -> f64,
    {
        // F0. prepare data
        let n = force.len();
        let mut velocity = self.velocity.unwrap_or(Velocity(vec![0.0; n]));
        // caching displacement for memory performance
        let mut displacement = self.displacement.unwrap_or(Displacement(vec![0.0; n]));

        // MD. calculate displacement vectors based on a typical MD stepping algorithm
        // update the internal velocity
        velocity.take_md_step(&force_prev, &force, self.dt, self.scheme);
        displacement.take_md_step(&force, &velocity, self.dt, self.scheme);
        displacement.rescale(self.max_step);

        // perform line search
        let nls = if self.use_line_search {
            info!("line search for optimal step size using MoreThuente algorithm.");
            40
        } else {
            1
        };
        let ls = linesearch()
            .with_max_iterations(nls)
            .with_algorithm("MoreThuente");

        let mut step = 1.0;
        let phi = |trial_step: f64| {
            // restore position
            // problem.set_position(&x_prev);
            problem.revert();
            // update value and gradient at position `x`
            problem.take_line_step(&displacement, trial_step);
            step = trial_step;
            let fx = problem.value();

            (fx, displacement.vecdot(&problem.gradient()))
        };
        let _ = ls.find(phi).expect("ls");

        // save state
        force.vecncpy(problem.gradient());
        force_prev.vecncpy(problem.gradient_prev());

        // F1. calculate power for uphill/downhill check
        let downhill = force.vecdot(&velocity).is_sign_positive();

        // F2. adjust velocity
        velocity.adjust(force, self.alpha);

        // F3 & F4: check the direction: go downhill or go uphill
        if downhill {
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
            self.dt = self.dt_min.max(self.dt * self.f_dec);
            // reset alpha
            self.alpha = self.alpha_start;
            // reset step counter
            self.nsteps = 0;
            // reset velocity to zero
            velocity.reset();
        }

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
        let mut problem = CachedProblem::new(x, f);

        // Check convergence.
        // Make sure that the initial variables are not a minimizer.
        if problem.gradient().vec2norm() <= self.termination.max_gradient_norm {
            info!("already converged.");
            return;
        }

        // FIRE algorithm uses force instead of gradient
        let mut force = problem.gradient().to_vec();
        force.vecscale(-1.0);
        let mut force_prev = force.clone();
        for i in 1.. {
            self = self.propagate(&mut force_prev, &mut force, &mut problem);
            if let Some(ref displ) = self.displacement {
                let fx = problem.value();

                let progress = Progress {
                    x,
                    fx,
                    step: 1.0,
                    niter: i,
                    gx: &force,
                    gnorm: force.vec2norm(),
                    displacement: displ,
                    ncall: problem.ncalls(),
                };

                if let Some(ref mut stopping) = stopping {
                    if stopping.met(&progress) {
                        break;
                    }
                }

                if self.termination.met(&progress) {
                    info!("normal termination!");
                    break;
                }
            } else {
                unreachable!()
            }
        }
    }
}
// entry:1 ends here

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*displacement][displacement:2]]
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
        dt: f64,          // Δt
        scheme: MdScheme,
    ) {
        debug_assert!(
            velocity.len() == force.len()
            "the sizes of input vectors are different!"
        );

        self.0.veccpy(velocity);
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
            // Semi-implicit Euler (SE)
            //
            // D = dt * V
            MdScheme::SemiImplicitEuler => {
                self.0.vecscale(dt);
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
}

// Deref coercion: use Displacement as a normal vec
use std::ops::Deref;
impl Deref for Displacement {
    type Target = Vec<f64>;

    fn deref(&self) -> &Vec<f64> {
        &self.0
    }
}
// displacement:2 ends here

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*velocity][velocity:2]]
/// Represents the velocity vector
///
/// # Example
/// 
/// ```ignore
/// let v = Velocity(vec![0.1, 0.2, 0.3]);
/// v.reset();
/// assert_eq!(0.0, v[1]);
/// ```
///
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
                self.0.vecadd(force_this, dt);
            }
        }
    }
}
// velocity:2 ends here
