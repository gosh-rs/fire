// base

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*base][base:1]]
//! Implementation of the Fast-Inertial-Relaxation-Engine (FIRE) algorithm
//! 
//! References
//! ----------
//! * Bitzek, E. et al. Structural Relaxation Made Simple. Phys. Rev. Lett. 2006, 97 (17), 170201.
//! * http://users.jyu.fi/~pekkosk/resources/pdf/FIRE.pdf
//! * https://github.com/siesta-project/flos/blob/master/flos/optima/fire.lua

mod fire;

/// Create FIRE optimization interface
pub fn fire() -> fire::FIRE {
    fire::FIRE::default()
}

/// Create user defined monitor or termination criteria with a closure
pub fn monitor<G>(mut cb: G) -> Option<UserTermination<G>>
where
    G: FnMut(&Progress) -> bool,
{
    Some(UserTermination(cb))
}

pub use crate::fire::FIRE;
// base:1 ends here

// progress

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*progress][progress:1]]
/// Holds important progress data per iteration in minimization.
#[derive(Debug, Clone)]
pub struct Progress {
    pub fx: f64,
    pub niter: usize,
    pub gnorm: f64,
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
            max_gradient_norm: 0.1,
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
