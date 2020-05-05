use criterion::{criterion_group, Criterion, BenchmarkId};
use b2dp::{Eta, GeneratorOpenSSL,mechanisms::naive::naive_exponential_mechanism, mechanisms::laplace::clamped_laplace_mechanism};

fn utility_fn(x: &f64) -> f64 {
    (0.0-*x).abs()
}


fn run_mechanism(gamma: f64, optimize: bool) -> f64 {
    let eta = Eta::new(1,1,1).unwrap();
    let rng = GeneratorOpenSSL {};
    let result = clamped_laplace_mechanism(eta,-10.0,10.0,0.0,gamma,rng,optimize,false).unwrap();
    result
}

fn not_optimized(gamma: f64) -> f64 {
    run_mechanism(gamma, false)
}

fn optimized(gamma: f64) -> f64 {
    run_mechanism(gamma, true)
}

fn run_naive(gamma: f64) -> f64 {
    let epsilon = 1.3; //Eta::new(1,1,1).unwrap();
    let rng = GeneratorOpenSSL {};
    let mut outcomes: Vec<f64> = Vec::new();
    //outcomes.push(0);
    //let k: u32 = 1000; // outcome space size
    let mut x = -10.0;
    while x <= 10.0 {
        outcomes.push(x);
        x += gamma;
    }
    let result = naive_exponential_mechanism(epsilon, &outcomes, utility_fn, rng).unwrap();
    *result
}


fn bench_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Laplace");
    group.sample_size(10);
    for i in [1.0,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625].iter() {
        group.bench_with_input(BenchmarkId::new("Not Optimized", i), i, 
            |b, i| b.iter(|| not_optimized(*i)));
        group.bench_with_input(BenchmarkId::new("Optimized", i), i, 
           |b, i| b.iter(|| optimized(*i)));
        group.bench_with_input(BenchmarkId::new("Naive", i), i, 
            |b, i| b.iter(|| run_naive(*i)));
    }
    group.finish();
}

criterion_group!(benches, bench_sizes);
//criterion_main!(benches);