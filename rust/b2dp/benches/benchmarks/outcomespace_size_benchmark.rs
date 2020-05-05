use criterion::{criterion_group, Criterion, BenchmarkId};
use b2dp::{Eta, exponential_mechanism, GeneratorOpenSSL,mechanisms::naive::naive_exponential_mechanism};

fn utility_fn(x: &u32) -> f64 {
    *x as f64
}


fn run_mechanism(n: i64, optimize: bool) -> u32 {
    let eta = Eta::new(1,1,1).unwrap();
    let rng = GeneratorOpenSSL {};
    let mut outcomes: Vec<u32> = Vec::new();
    for i in 1..n {
        outcomes.push(i as u32);
    }
    let result = exponential_mechanism(eta, &outcomes, utility_fn, 0, n as i64, n as u32, rng, optimize, false).unwrap();
    *result
}

fn not_optimized(n: i64) -> u32 {
    run_mechanism(n, false)
}

fn optimized(n: i64) -> u32 {
    run_mechanism(n, true)
}

fn run_naive(n: i64) -> u32 {
    let epsilon = 1.3; 
    let rng = GeneratorOpenSSL {};
    let mut outcomes: Vec<u32> = Vec::new();
    
    for i in 1..n {
        outcomes.push(i as u32);
    }
    let result = naive_exponential_mechanism(epsilon, &outcomes, utility_fn, rng).unwrap();
    *result
}


fn bench_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Outcome Space Size");
    group.sample_size(10);
    for i in [100, 1000, 5000, 10000, 15000, 20000, 25000, 50000, 75000].iter() {
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