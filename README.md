# Benchmark

Benchmarks for the linear-time risk-measure algorithms in
[RiskMeasures.jl](https://github.com/marekpetrik/RiskMeasures.jl). The package
times CVaR, qCVaR, VaR, qVaR, TVaR, qTVaR and the plain expectation on both
randomly generated and stock-derived distributions.

## Installation

Requires Julia 1.12 or newer.

Install directly from the GitHub repository. The package lives in the
`Benchmark` subdirectory, so pass `subdir`:

```julia
using Pkg
Pkg.add(url="https://github.com/marekpetrik/paper_lineartime_risk.git", subdir="Benchmark")
```

Alternatively, from the Pkg REPL (press `]`):

```
pkg> add https://github.com/marekpetrik/paper_lineartime_risk.git:Benchmark
```

## Usage

Both exported functions return measured timings (in milliseconds) that you can
save to CSV.

Benchmark on randomly generated distributions. `benchmark_random` returns a
`Dict` keyed by distribution name (`"uniform"` and `"sparse"`), so write one CSV
per distribution:

```julia
using Benchmark, CSV

results = benchmark_random()
for (dist, df) in results
    CSV.write("benchmark_random_$dist.csv", df)
end
```

The default sizes (1e6 to 1e7) take a while. To run a quick test, pass
small `trials`, `start`, `step` and `stop` values:

```julia
using Benchmark, CSV

results = benchmark_random(trials=3, start=1000, step=1000, stop=3000)
for (dist, df) in results
    CSV.write("benchmark_random_$dist.csv", df)
end
```

Benchmark on stock-derived distributions (uses `data/spy_data.csv`).
`benchmark_stocks` returns a single `DataFrame`:

```julia
using Benchmark, CSV

df = benchmark_stocks()
CSV.write("benchmark_stocks.csv", df)
```

Again, to run a quick test, pass parameters to benchmark_stocks as follows.

```julia
using Benchmark, CSV

df = benchmark_stocks(trials = 3, window = 5)
CSV.write("benchmark_stocks.csv", df)
```

## Plotting

`plot_result` reads a saved CSV back in and returns a plot comparing the slow
and fast methods. Use `savefig` to write the figure to a PDF:

```julia
using Benchmark, Plots

plt = plot_result("benchmark_stocks.csv")
savefig(plt, "benchmark_stocks.pdf")
```
