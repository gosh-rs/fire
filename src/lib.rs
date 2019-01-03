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

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*docs][docs:2]]
use self::common::*;

/// Represents an optimization problem with cache for avoiding unnecessary
/// function re-evaluations.
///
/// # Examples
///
/// ```ignore
/// let mut x = vec![0.0; 5];
/// let mut prb = CachedProblem::new(&x, f);
/// let d = [0.2; 5];
/// prb.take_line_step(&d, 1.0);
/// let fx = prb.value();
/// let gx = prb.gradient();
/// ```
// docs:2 ends here

// cached

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*cached][cached:1]]
#[derive(Clone, Debug)]
pub struct CachedProblem<E>
where
    E: FnMut(&[f64], &mut [f64]) -> f64,
{
    f: E,
    x: Vec<f64>,
    fx: Option<f64>,
    gx: Option<Vec<f64>>,

    epsilon: f64,
    neval: usize,

    // cache previous point
    x_prev: Vec<f64>,
    fx_prev: Option<f64>,
    gx_prev: Option<Vec<f64>>,
}

impl<E> CachedProblem<E>
where
    E: FnMut(&[f64], &mut [f64]) -> f64,
{
    /// Construct a CachedProblem
    ///
    /// # Parameters
    ///
    /// * x: initial position
    /// * f: a closure for function evaluation of value and gradient.
    pub fn new(x: &[f64], f: E) -> Self {
        CachedProblem {
            f,
            epsilon: 1e-8,
            x: x.to_vec(),
            fx: None,
            gx: None,
            neval: 0,
            x_prev: x.to_vec(),
            fx_prev: None,
            gx_prev: None,
        }
    }

    /// The number of function calls
    pub fn ncalls(&self) -> usize {
        self.neval
    }

    /// Update position `x` at a prescribed displacement and step size.
    ///
    /// x += step * displ
    pub fn take_line_step(&mut self, displ: &[f64], step: f64) {
        // position changed
        if step * displ.vec2norm() > self.epsilon {
            // update position vector with displacement
            self.x.vecadd(displ, step);
            self.fx = None;
            self.gx = None;
        }
    }

    /// evaluate function value and gradient at current position
    fn eval(&mut self) -> (f64, &[f64]) {
        let n = self.x.len();

        let gx: &mut [f64] = self.gx.get_or_insert(vec![0.0; n]);
        let v = (self.f)(&self.x, gx);
        self.fx = Some(v);
        self.neval += 1;

        // update previous point
        self.fx_prev = self.fx;
        self.x_prev = self.x.clone();
        self.gx_prev = Some(gx.to_vec());

        (v, gx)
    }

    /// Return function value at current position.
    ///
    /// The function will be evaluated when necessary.
    pub fn value(&mut self) -> f64 {
        match self.fx {
            // found cached value.
            Some(v) => v,
            // first time calculation
            None => {
                let (fx, _) = self.eval();
                fx
            }
        }
    }

    /// Return function value at previous point
    pub fn value_prev(&self) -> f64 {
        match self.fx_prev {
            Some(fx) => fx,
            None => panic!("not evaluated yet"),
        }
    }

    pub fn gradient_prev(&self) -> &[f64] {
        match self.gx_prev {
            Some(ref gx) => gx,
            None => panic!("not evaluated yet"),
        }
    }

    /// Return a reference to gradient at current position.
    ///
    /// The function will be evaluated when necessary.
    pub fn gradient(&mut self) -> &[f64] {
        match self.gx {
            // found cached value.
            Some(ref gx) => gx,
            // first time calculation
            None => {
                let (_, gx) = self.eval();
                gx
            }
        }
    }

    /// Return a reference to current position vector.
    pub fn position(&self) -> &[f64] {
        &self.x
    }

    /// Revert to previous point
    pub fn revert(&mut self) {
        self.fx = self.fx_prev;
        self.x.veccpy(&self.x_prev);
        self.gx = self.gx_prev.clone();
    }
}
// cached:1 ends here
