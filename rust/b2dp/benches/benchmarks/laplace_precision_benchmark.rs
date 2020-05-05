use criterion::{criterion_group, Criterion, BenchmarkId};
use b2dp::{Eta, GeneratorOpenSSL, mechanisms::laplace::clamped_laplace_mechanism};


fn run_mechanism(gamma: f64, precision: bool) -> f64 {
    let eta = Eta::new(1,1,1).unwrap();
    let optimize = false;
    let rng = GeneratorOpenSSL {};
    let result = clamped_laplace_mechanism(eta,-10.0,10.0,0.0,gamma,rng,optimize,precision).unwrap();
    result
}

fn not_optimized(gamma: f64) -> f64 {
    run_mechanism(gamma, false)
}

fn optimized(gamma: f64) -> f64 {
    run_mechanism(gamma, true)
}


fn bench_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Laplace Precision");
    group.sample_size(10);
    for i in [1.0,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625].iter() {
        group.bench_with_input(BenchmarkId::new("Theoretical", i), i, 
            |b, i| b.iter(|| not_optimized(*i)));
        group.bench_with_input(BenchmarkId::new("Empirical", i), i, 
           |b, i| b.iter(|| optimized(*i)));
    }
    group.finish();
}

criterion_group!(benches, bench_sizes);
//criterion_main!(benches);