# Benchmark

Benchmarks for the linear-time risk-measure algorithms in
[RiskMeasures.jl](https://github.com/marekpetrik/RiskMeasures.jl). The package
times CVaR, qCVaR, VaR, qVaR, TVaR, qTVaR and the plain expectation on both
randomly generated and stock-derived distributions.

## Installation

Requires Julia 1.12 or newer.

```julia
using Pkg
Pkg.develop(path="/path/to/Benchmark")  # adjust to where this package lives
Pkg.instantiate()                       # install the dependencies
```

Alternatively, from the Pkg REPL (press `]`) inside the package directory:

```
pkg> activate .
pkg> instantiate
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

Benchmark on stock-derived distributions (uses `data/spy_data.csv`).
`benchmark_stocks` returns a single `DataFrame`:

```julia
using Benchmark, CSV

df = benchmark_stocks()
CSV.write("benchmark_stocks.csv", df)
```
