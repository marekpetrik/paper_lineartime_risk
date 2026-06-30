# Benchmark

Benchmarks for the linear-time risk-measure algorithms in
[RiskMeasures.jl](https://github.com/RiskAverseRL/RiskMeasures.jl). This project
times CVaR, qCVaR, VaR, qVaR, TVaR, qTVaR and the plain expectation on both
randomly generated and stock-derived distributions.

This is a plain Julia project (a script plus a `Project.toml` environment), not
an installable package.

## Setup

Requires Julia 1.12 or newer.

Clone the repository and, from its root directory, activate the project
environment and install its dependencies:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Equivalently, from the Pkg REPL (press `]`):

```
pkg> activate .
pkg> instantiate
```

The functions live in `benchmark.jl`. Load them with `include`:

```julia
include("benchmark.jl")
```

You can also start Julia with the environment already active:

```sh
julia --project=. benchmark.jl
```

## Reproducing the paper

To reproduce the results and plots from the matching paper, run `runall.jl`
from the project root with the environment active:

```sh
julia --project=. runall.jl
```

This runs the full benchmark suite and writes the result CSVs. To avoid
overwriting existing results, it errors out if any of the target CSV files
already exist. From the saved CSVs you can generate the figures as described
under [Plotting](#plotting).

## Usage

After `include("benchmark.jl")`, the functions `benchmark_random`, `benchmark_stocks` and `plot_result` are available in your session. Both benchmarking functions return measured timings (in milliseconds) that you can save to CSV.

Benchmark on randomly generated distributions. `benchmark_random` returns a `Dict` keyed by distribution name (`"uniform"` and `"sparse"`), so write one CSV per distribution:

```julia
using CSV

results = benchmark_random()
for (dist, df) in results
    CSV.write("benchmark_random_$dist.csv", df)
end
```

The default sizes (1e6 to 1e7) take a while. To run a quick test, pass small `trials`, `start`, `step` and `stop` values:
```julia
using CSV

results = benchmark_random(trials=3, start=1000, step=1000, stop=3000)
for (dist, df) in results
    CSV.write("benchmark_random_$dist.csv", df)
end
```

Benchmark on stock-derived distributions (uses `data/spy_data.csv`).
`benchmark_stocks` returns a single `DataFrame`:

```julia
using CSV

df = benchmark_stocks()
CSV.write("benchmark_stocks.csv", df)
```

Again, to run a quick test, pass parameters to `benchmark_stocks` as follows:

```julia
using CSV

df = benchmark_stocks(trials = 3, window = 5)
CSV.write("benchmark_stocks.csv", df)
```

## Plotting

`plot_result` reads a saved CSV back in and returns a plot comparing the slow
and fast methods. Use `savefig` to write the figure to a PDF:

```julia
using Plots

plt = plot_result("benchmark_stocks.csv")
savefig(plt, "benchmark_stocks.pdf")
```
