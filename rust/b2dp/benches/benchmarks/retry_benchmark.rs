use criterion::{criterion_group, Criterion, BenchmarkId};
use b2dp::{Eta, exponential_mechanism, GeneratorOpenSSL, mechanisms::exponential::ExponentialOptions};

fn utility_fn(x: &u32) -> f64 {
    *x as f64
}


fn run_mechanism(num_retries: u32) -> u32 {
    let eta = Eta::new(1,1,1).unwrap();
    let rng = GeneratorOpenSSL {};
    let mut outcomes: Vec<u32> = Vec::new();
    let n = 1000;
    let optimize = false;
    for i in 1..n {
        outcomes.push(i as u32);
    }
    let options = ExponentialOptions { min_retries: num_retries, optimized_sample: optimize, empirical_precision: false};
    let result = exponential_mechanism(eta, &outcomes, utility_fn, 0, n as i64, n as u32, rng, options).unwrap();
    *result
}




fn bench_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Retry");
    group.sample_size(10);
    for i in [1, 2, 3, 5, 10, 20].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(i) , i, 
            |b, i| b.iter(|| run_mechanism(*i)));
    }
    group.finish();
}

criterion_group!(benches, bench_sizes);
//criterion_main!(benches);