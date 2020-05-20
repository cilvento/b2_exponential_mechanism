use criterion::{criterion_group, Criterion, BenchmarkId};
use b2dp::{Eta, exponential_mechanism, GeneratorOpenSSL, mechanisms::exponential::ExponentialOptions};

fn utility_fn(x: &u32) -> f64 {
    *x as f64
}


fn run_mechanism(num_retries: u32, weight_low: bool) -> u32 {
    let eta = Eta::new(1,1,1).unwrap();
    let rng = GeneratorOpenSSL {};
    let mut outcomes: Vec<u32> = Vec::new();
    let n = 256;
    let optimize = false;
    if weight_low {outcomes.push(0);}
    else {  outcomes.push(1);}
    for _i in 1..n {
        outcomes.push(1);
    }
    let options = ExponentialOptions { min_retries: num_retries, optimized_sample: optimize, empirical_precision: false};
    let result = exponential_mechanism(eta, &outcomes, utility_fn, 0, n as i64, n as u32, rng, options).unwrap();
    *result
}




fn bench_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Timing Channel Demo");
    
    for i in 1..21 {
        group.bench_with_input(BenchmarkId::new("HigherWeight", i) , &i, 
            |b, i| b.iter(|| run_mechanism(*i, false)));
        group.bench_with_input(BenchmarkId::new("LowerWeight", i) , &i, 
            |b, i| b.iter(|| run_mechanism(*i, true)));
    }
    group.finish();
}

criterion_group!(benches, bench_sizes);
//criterion_main!(benches);