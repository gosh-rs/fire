// rosenbrock
// :PROPERTIES:
// :header-args: :tangle tests/rosenbrock.rs
// :END:

// [[file:~/Workspace/Programming/rust-scratch/fire/fire.note::*rosenbrock][rosenbrock:1]]
#[test]
fn test() {
    use fire::fire;
    use fire::monitor;
    use fire::GradientBasedMinimizer;

    // 1. Initialize data
    const N: usize = 100;

    let mut x = [0.0 as f64; N];
    for i in (0..N).step_by(2) {
        x[i] = -1.2;
        x[i + 1] = 1.0;
    }

    // 2. Defining how to evaluate function and gradient
    let evaluate = |x: &[f64], gx: &mut [f64]| {
        let n = x.len();

        let mut fx = 0.0;
        for i in (0..n).step_by(2) {
            let t1 = 1.0 - x[i];
            let t2 = 10.0 * (x[i + 1] - x[i] * x[i]);
            gx[i + 1] = 20.0 * t2;
            gx[i] = -2.0 * (x[i] * gx[i + 1] + t1);
            fx += t1 * t1 + t2 * t2;
        }

        fx
    };

    let m = monitor(|prgr| {
        println!(
            "niter = {:5}: fx = {:-10.4}, gnorm = {:-10.4}",
            prgr.niter, prgr.fx, prgr.gnorm
        );
        false
    });

    fire()
        .with_max_step(0.20)
        .with_max_cycles(500)
        .minimize_alt(&mut x, evaluate, m);
}
// rosenbrock:1 ends here
