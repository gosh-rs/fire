// base

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*base][base:1]]
//! Implementation of the Fast-Inertial-Relaxation-Engine (FIRE) algorithm
//! 
//! References
//! ----------
//! * Bitzek, E. et al. Structural Relaxation Made Simple. Phys. Rev. Lett. 2006, 97 (17), 170201.
//! * http://users.jyu.fi/~pekkosk/resources/pdf/FIRE.pdf
//! * https://github.com/siesta-project/flos/blob/master/flos/optima/fire.lua
pub mod line;
pub mod lj;

pub(crate) mod common {
    pub use quicli::prelude::*;
    pub type Result<T> = ::std::result::Result<T, Error>;
}

/// Create FIRE optimization interface
pub fn fire() -> fire::FIRE {
    fire::FIRE::default()
}

/// Create user defined monitor or termination criteria with a closure
pub fn monitor<G>(cb: G) -> Option<UserTermination<G>>
where
    G: FnMut(&Progress) -> bool,
{
    Some(UserTermination(cb))
}

mod fire;
pub use crate::fire::FIRE;
// base:1 ends here

// progress

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*progress][progress:1]]
use vecfx::*;

/// Important iteration data in minimization, useful for progress monitor or
/// defining artificial termination criteria.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Progress<'a> {
    /// Cached value of gradient vector norm.
    gnorm: f64,

    /// The number of iteration in optimization loop.
    niter: usize,

    /// The actual number of function value/gradient calls. This number can be
    /// larger than `niter`.
    ncall: usize,

    /// The line-search step used for this iteration.
    pub step: f64,

    /// displacement vector over input variables vector
    displacement: &'a [f64],

    /// Reference to current input variables vector
    x: &'a [f64],

    /// Reference to current gradient vector of f(x): 'f(x)
    gx: &'a [f64],

    /// The current value of the objective function.
    fx: f64,
}

impl<'a> Progress<'a> {
    /// Return a reference to gradient vector.
    pub fn gradient(&self) -> &'a [f64] {
        self.gx
    }

    /// Return current function value.
    pub fn value(&self) -> f64 {
        self.fx
    }

    /// Return a reference to displacement vector.
    pub fn displacement(&self) -> &'a [f64] {
        self.displacement
    }

    /// Print a summary about progress.
    pub fn report(&self) {
        println!(
            "niter = {:5}, ncall= {:5}: fx = {:-10.4}, gnorm = {:-10.4}, dnorm = {:-10.4} step = {:-10.4}",
            self.niter,
            self.ncall,
            self.fx,
            self.gnorm,
            self.displacement.vec2norm(),
            self.step,
        );
    }
}
// progress:1 ends here

// termination interface

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*termination%20interface][termination interface:1]]
/// Define termination conditions
pub trait TerminationCriteria {
    fn met(&mut self, progress: &Progress) -> bool;
}

/// For user defined termination criteria
#[derive(Debug, Clone)]
pub struct UserTermination<G>(pub G)
where
    G: FnMut(&Progress) -> bool;

impl<G> TerminationCriteria for UserTermination<G>
where
    G: FnMut(&Progress) -> bool,
{
    fn met(&mut self, progress: &Progress) -> bool {
        (self.0)(progress)
    }
}

/// Termination criteria
#[derive(Debug, Clone)]
pub(crate) struct Termination {
    /// The maximum number of optimization cycles.
    pub max_cycles: usize,
    /// The max allowed gradient norm
    pub max_gradient_norm: f64,
}

impl Default for Termination {
    fn default() -> Self {
        Termination {
            max_cycles: 0,
            max_gradient_norm: 0.01,
        }
    }
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
// termination interface:1 ends here

// minimizing interface

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*minimizing%20interface][minimizing interface:1]]
/// Common interfaces for structure relaxation
pub trait GradientBasedMinimizer: Sized {
    fn minimize<E, G>(mut self, x: &mut [f64], mut f: E)
    where
        E: FnMut(&[f64], &mut [f64]) -> f64,
        G: TerminationCriteria,
    {
        self.minimize_alt::<E, G>(x, f, None)
    }

    fn minimize_alt<E, G>(mut self, x: &mut [f64], mut f: E, mut stopping: Option<G>)
    where
        E: FnMut(&[f64], &mut [f64]) -> f64,
        G: TerminationCriteria;
}
// minimizing interface:1 ends here

// problem interface

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*problem%20interface][problem interface:1]]
use self::common::*;

/// Optimization problem
pub trait Problem {
    fn new<E>(x: &mut [f64], f: E) -> Self
    where
        E: FnMut(&[f64], &mut [f64]) -> Result<f64>;

    /// Return search direction
    fn search_direction(&self) -> &[f64];

    /// Return search direction in mutable
    fn search_direction_mut(&mut self) -> &mut [f64];

    /// Define how to evaluate function value and gradient
    fn evaluate(&mut self) -> Result<()>;

    /// Return gradient norm
    fn gnorm(&self) -> f64;

    /// Rever to previous step
    fn revert(&mut self);

    /// Return initial step size
    fn initial_step(&self) -> f64;

    /// Take a line step along search direction
    fn take_line_step(&mut self, stp: f64);

    /// Compute the initial gradient in the search direction.
    fn dg(&self) -> f64;

    /// Compute the gradient in the search direction without sign checking.
    fn dg_uncheck(&self) -> f64;

    /// Return total number of evaluations.
    fn nevaluations(&self) -> usize;
}
// problem interface:1 ends here
