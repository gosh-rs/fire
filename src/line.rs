// header

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*header][header:1]]
use crate::common::*;
// header:1 ends here

// base

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*base][base:1]]
#[derive(Clone, Debug, PartialEq)]
pub enum LineSearchCondition {
    /// The sufficient decrease condition.
    Armijo,
    Wolfe,
    StrongWolfe,
}

pub trait LineSearch<E>
where
    E: FnMut(f64, &mut f64) -> f64,
{
    /// Given initial step size and phi function, returns an satisfactory step
    /// size.
    fn find(&mut self, step: f64, phi: E) -> Result<f64>;
}

#[derive(Clone, Debug)]
pub struct Backtracking {
    /// A parameter to control the accuracy of the line search routine.
    ///
    /// The default value is 1e-4. This parameter should be greater
    /// than zero and smaller than 0.5.
    ftol: f64,

    /// A parameter to control the accuracy of the line search routine.
    ///
    /// The default value is 0.9. If the function and gradient evaluations are
    /// inexpensive with respect to the cost of the iteration (which is
    /// sometimes the case when solving very large problems) it may be
    /// advantageous to set this parameter to a small value. A typical small
    /// value is 0.1. This parameter shuold be greater than the ftol parameter
    /// (1e-4) and smaller than 1.0.
    gtol: f64,

    /// The maximum number of trials for the line search.
    ///
    /// This parameter controls the number of function and gradients evaluations
    /// per iteration for the line search routine. The default value is 40. Set
    /// this value to 0, will completely disable line search.
    ///
    max_iterations: usize,

    /// The factor to increase step size
    fdec: f64,
    /// The factor to decrease step size
    finc: f64,

    /// The minimum step of the line search routine.
    ///
    /// The default value is 1e-20. This value need not be modified unless the
    /// exponents are too large for the machine being used, or unless the
    /// problem is extremely badly scaled (in which case the exponents should be
    /// increased).
    min_step: f64,

    /// The maximum step of the line search.
    ///
    /// The default value is 1e+20. This value need not be modified unless the
    /// exponents are too large for the machine being used, or unless the
    /// problem is extremely badly scaled (in which case the exponents should be
    /// increased).
    max_step: f64,

    /// Inexact line search condition
    condition: LineSearchCondition,
}

impl Default for Backtracking {
    fn default() -> Self {
        Backtracking {
            ftol: 1e-4,
            gtol: 0.9,
            fdec: 0.5,
            finc: 2.1,

            min_step: 1e-20,
            max_step: 1e20,

            max_iterations: 40,
            condition: LineSearchCondition::StrongWolfe,
        }
    }
}

impl<E> LineSearch<E> for Backtracking
where
    E: FnMut(f64, &mut f64) -> f64,
{
    /// # Parameters
    ///
    /// * stp: initial step size
    /// * phi: callback function for evaluating value and gradient along searching direction
    fn find(&mut self, stp: f64, mut phi: E) -> Result<f64> {
        use self::LineSearchCondition::*;

        let mut dginit = 0.0;
        let phi0 = phi(0.0, &mut dginit);
        let dgtest = self.ftol * dginit;

        let mut step = stp;
        let mut dg = 0.0;
        for k in 1..self.max_iterations {
            // Evaluate the function and gradient values along search direction.
            let phi_k = phi(step, &mut dg);

            let width = if phi_k > phi0 + step * dgtest {
                self.fdec
            } else if self.condition == Armijo {
                // The sufficient decrease condition.
                // Exit with the Armijo condition.
                return Ok(step)
            } else {
                // Check the Wolfe condition.
                if dg < self.gtol * dginit {
                    self.finc
                } else if self.condition == Wolfe {
                    // Exit with the regular Wolfe condition.
                    return Ok(step);
                } else if dg > -self.gtol * dginit {
                    self.fdec
                } else {
                    return Ok(step);
                }
            };

            step *= width;

            // The step is the minimum value.
            if step < self.min_step {
                bail!("The line-search step became smaller than LineSearch::min_step.");
            }
            // The step is the maximum value.
            if step > self.max_step {
                bail!("The line-search step became larger than LineSearch::max_step.");
            }
        }

        warn!("max allowed line searches reached!");

        Ok(step)
    }
}
// base:1 ends here
