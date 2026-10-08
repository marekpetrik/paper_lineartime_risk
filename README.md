# Benchmark

Benchmarks for the linear-time risk-measure algorithms in
[RiskMeasures.jl](https://github.com/RiskAverseRL/RiskMeasures.jl). This project
times CVaR, qCVaR, VaR, qVaR, TVaR, qTVaR and the plain expectation on
randomly generated and stock-derived distributions, and within risk-averse
(nested) value iteration on an inventory MDP.

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

The [MDPs.jl](https://github.com/RiskAverseRL/MDPs.jl) dependency is not
registered; `instantiate` installs it from GitHub through the `[sources]` entry
in `Project.toml`.

The random-variable benchmarks, plotting and table functions live in
`benchmark.jl`; the value iteration benchmark lives in
`risk_averse_value_iteration.jl`. Load them with `include`:

```julia
include("benchmark.jl")
include("risk_averse_value_iteration.jl")
```

You can also start Julia with the environment already active:

```sh
julia --project=. benchmark.jl
```

## Reproducing the paper

To reproduce the results, plots and tables from the matching paper, run
`runall.jl` from the project root with the environment active:

```sh
julia --project=. -t auto runall.jl
```

(`-t auto` lets value iteration parallelize over states.)

This runs the full benchmark suite (random variables, stocks and value
iteration) and writes the result CSVs. To avoid overwriting existing results,
any benchmark whose CSV file already exists is skipped. It then generates the
figures and LaTeX tables from the saved CSVs as described under
[Plotting](#plotting):

| Benchmark        | CSV                             | Plot  | Tables                    |
|------------------|---------------------------------|-------|---------------------------|
| random uniform   | `benchmark_random_uniform.csv`  | `.pdf`|                           |
| random sparse    | `benchmark_random_sparse.csv`   | `.pdf`|                           |
| random small     | `benchmark_random_small.csv`    | `.pdf`| `_mean.tex`, `_std.tex`   |
| stocks           | `benchmark_stocks.csv`          |       | `_mean.tex`, `_std.tex`   |
| value iteration  | `benchmark_value_iteration.csv` | `.pdf`| `_mean.tex`, `_std.tex`   |

To regenerate a benchmark, delete its CSV and rerun `runall.jl`.


## Usage

After the `include`s above, the functions `benchmark_random`,
`benchmark_stocks`, `benchmark_value_iteration`, `plot_result` and
`generate_tables` are available in your session. The benchmarking functions
return measured timings (in milliseconds) that you can save to CSV.

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

Benchmark risk-averse nested value iteration on inventory MDPs of increasing
size. `benchmark_value_iteration` returns a single `DataFrame` with the same
columns, where `n` is the number of states:

```julia
using CSV

df = benchmark_value_iteration()
CSV.write("benchmark_value_iteration.csv", df)
```

For a quick test, use fewer trials and time steps (keep the state counts
spanning a power of 10, or the log-scale plot has no axis ticks):

```julia
using CSV

df = benchmark_value_iteration(trials = 2, states = [50, 100], T = 2)
CSV.write("benchmark_value_iteration.csv", df)
```

## Plotting

`plot_result` reads a saved CSV back in and returns a plot comparing the slow
and fast methods. Use `savefig` to write the figure to a PDF:

```julia
using Plots

plt = plot_result("benchmark_random_uniform.csv")
savefig(plt, "benchmark_random_uniform.pdf")
```

Pass `xlabel` to change the x-axis label, e.g. for the value iteration results:

```julia
plt = plot_result("benchmark_value_iteration.csv"; xlabel = "Number of States")
savefig(plt, "benchmark_value_iteration.pdf")
```

`generate_tables` writes LaTeX tables with the mean runtimes and 95% confidence
interval half-widths per `n` next to the CSV (`<name>_mean.tex` and
`<name>_std.tex`):

```julia
generate_tables("benchmark_value_iteration.csv")
```
