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
    velocities: Option<Vec<f64>>,

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
