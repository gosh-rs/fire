// lj.rs
// :PROPERTIES:
// :header-args: :tangle src/lj.rs
// :END:

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*lj.rs][lj.rs:1]]
use vecfx::*;

#[derive(Clone, Copy, Debug)]
pub struct LennardJones {
    /// Energy constant of the Lennard-Jones potential
    pub epsilon: f64,
    /// Distance constant of the Lennard-Jones potential
    pub sigma: f64,
}

impl Default for LennardJones {
    fn default() -> Self {
        LennardJones {
            epsilon: 1.0,
            sigma: 1.0,
        }
    }
}

impl LennardJones {
    // vij
    fn pair_energy(&self, r: f64) -> f64 {
        let s6 = f64::powi(self.sigma / r, 6);
        4.0 * self.epsilon * (f64::powi(s6, 2) - s6)
    }

    // dvij
    fn pair_gradient(&self, r: f64) -> f64 {
        let s6 = f64::powi(self.sigma / r, 6);

        24.0 * self.epsilon * (s6 - 2.0 * f64::powi(s6, 2)) / r
    }

    /// Evaluate energy and forces
    pub fn evaluate(&self, positions: &[[f64; 3]], forces: &mut [[f64; 3]]) -> f64 {
        let n = positions.len();
        debug_assert_eq!(n, forces.len(), "positions.len() != forces.len()");

        // initialize with zeros
        let mut energy = 0.0;
        for i in 0..n {
            for j in 0..3 {
                forces[i][j] = 0.0;
            }
        }

        // calculate energy and forces
        for i in 0..n {
            for j in 0..i {
                let r = positions[i].vecdist(&positions[j]);
                energy += self.pair_energy(r);

                let g = self.pair_gradient(r);
                for k in 0..3 {
                    let dr = positions[j][k] - positions[i][k];
                    forces[i][k] += 1.0 * g * dr / r;
                    forces[j][k] += -1.0 * g * dr / r;
                }
            }
        }

        energy
    }
}
// lj.rs:1 ends here
