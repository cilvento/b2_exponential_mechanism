use criterion::criterion_main;

mod benchmarks;

criterion_main! {
    benchmarks::laplace_precision_benchmark::benches,
    benchmarks::precision_benchmark::benches,
    benchmarks::laplace_benchmark::benches,
    benchmarks::utility_range_benchmark::benches,
    benchmarks::outcomespace_size_benchmark::benches,
    benchmarks::retry_benchmark::benches,
    benchmarks::timingchannel_benchmark::benches,
}