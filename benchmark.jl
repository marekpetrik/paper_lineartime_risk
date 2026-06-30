# Benchmarks for the linear-time risk-measure algorithms in RiskMeasures.jl.
#
# This is a plain script (not a package). Activate the project environment and
# `include` this file to get `benchmark_random`, `benchmark_stocks` and
# `plot_result` in your session:
#
#     julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
#     julia> include("benchmark.jl")
#
# See README.md for full usage.

include(joinpath(@__DIR__, "worstcasel1.jl"))

using Base.Threads
using Dates
using DataFrames
using CSV
using Random
using ProgressBars
using Distributions
using RiskMeasures
using Statistics
using Plots
using PGFPlotsX

# --- Plotting / RNG setup (run once when this file is included) ---
Random.seed!(1234)

# Use the PGFPlotsX backend (LaTeX) and set the font to LMRoman (Latin Modern Roman)
pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{lmodern}")
default(
    fontfamily = "Latin Modern Roman",
    titlefontsize = 18,
    guidefontsize = 18,
    tickfontsize = 18,
    legendfontsize = 18,
    size = (830, 600),
    grid = true,
)



"""
    run_one_experiment(x, p, α)

Run one experiment with random variable *x* with probability distribution
*p* and risk level *α* and return the results.

# Returns:
  NamedTuple of measured times (in milliseconds) for each algorithm.
"""
function run_one_experiment(x, p, α)
    # --- CVaR ---
    tmpx = deepcopy(x)
    tmpp = deepcopy(p)
    local start = time_ns()
    slow_cvar_result = RiskMeasures.CVaR(tmpx, tmpp, α, check_inputs=false, fast=false).value
    slow_time = (time_ns() - start) * 1e-6
    tmpx = deepcopy(x)
    tmpp = deepcopy(p)
    start = time_ns()
    fast_cvar_result = RiskMeasures.CVaR(tmpx, tmpp, α, check_inputs=false, fast=true).value
    fast_time = (time_ns() - start) * 1e-6
    # --- VaR ---
    tmpx = deepcopy(x)
    tmpp = deepcopy(p)
    start = time_ns()
    RiskMeasures.VaR(tmpx, tmpp, α, check_inputs=false, fast=false).value
    var_time = (time_ns() - start) * 1e-6
    tmpx = deepcopy(x)
    tmpp = deepcopy(p)
    start = time_ns()
    RiskMeasures.VaR(tmpx, tmpp, α, check_inputs=false, fast=true).value
    qvar_time = (time_ns() - start) * 1e-6
    # --- TVaR ---
    tmpx = deepcopy(x)
    tmpp = deepcopy(p)
    start = time_ns()
    worstcase_l1(tmpx, tmpp, 2*α)
    tvar_time = (time_ns() - start) * 1e-6
    tmpx = deepcopy(x)
    tmpp = deepcopy(p)
    start = time_ns()
    choquet_ews(tmpx, tmpp, choquet_ews_tvar(α))
    qtvar_time = (time_ns() - start) * 1e-6
    # --- Expect ---
    start = time_ns()
    expectation = sum(tmpx .* tmpp)
    expectation_time = (time_ns() - start) * 1e-6

    # check correctness
    δ = abs(slow_cvar_result - fast_cvar_result)
    if δ >= 1e-6
        println("Regular: $slow_cvar_result, Fast: $fast_cvar_result, diff: $δ")
        error("Results are not equal!")
    end

    (slow_cvar_time=slow_time, fast_cvar_time=fast_time, var_time=var_time,
     qvar_time=qvar_time, tvar_time=tvar_time, qtvar_time=qtvar_time,
     expectation_time=expectation_time)
end

function p_gen_func(dist)
    if dist == "uniform"
        return n -> fill(1 / n, n)
    elseif dist == "sparse"
        return n -> begin
            p = zeros(Float64, n)
            inds = unique(rand(1:n, Int(ceil(log(n)))))
            p[inds] .= 1 / length(inds)
            p
        end
  else
        error("Unknown distribution")
    end
end

"""
    benchmark_random(; trials = 10, start = Int(1e6), step = Int(1e6), stop = Int(1e7))

Benchmark the risk-measure algorithms on randomly generated distributions.

For both `"uniform"` and `"sparse"` probability distributions, this runs each
algorithm (CVaR, qCVaR, VaR, qVaR, TVaR, qTVaR and the plain expectation) on
random variables whose size ranges from `start` to `stop` in increments of
`step`, repeating each size `trials` times, and records the running time of each.

# Arguments
  - `trials`: number of repetitions per problem size.
  - `start`, `step`, `stop`: smallest size, increment, and largest size of the
    random variables (number of samples).

# Returns
A `Dict` keyed by distribution name (`"uniform"` and `"sparse"`). Each value is a
`DataFrame` with one row per trial and columns `n`, `cvar`, `qcvar`, `var`,
`qvar`, `tvar`, `qtvar` and `expectation` holding the measured times in
milliseconds.
"""
function benchmark_random(; trials = 10, start = Int(1e6), step = Int(1e6), stop = Int(1e7))
    println("Starting experiments")
    results = Dict()
    for dist in ["uniform", "sparse"]
        println("Running experiments for $dist")
        # returns a function that takes n and returns a probability distribution
        p_f = p_gen_func(dist)
        # One experiment is running the CVaR and qCVaR algorithms and collecting the time taken.
        experiments = Int.(ceil.(range(start=start, stop=stop, step=step)))
        # need to "multiply" experiments vector by trials
        experiments = hcat(repeat(experiments, 1, trials)'...)[:]
        len = length(experiments)
        qcvar_results = zeros(Float64, len)
        cvar_results = zeros(Float64, len)
        var_results = zeros(Float64, len)
        qvar_results = zeros(Float64, len)
        tvar_results = zeros(Float64, len)
        qtvar_results = zeros(Float64, len)
        expectation_results = zeros(Float64, len)
        for i ∈ ProgressBar(1:len)
            GC.enable(false)
            n = experiments[i]
            x = rand(Float64, n) .* 100
            p = p_f(n)
            α = 0.95 + 1e-6
            c, qc, v, qv, t, qt, e = run_one_experiment(x, p, α)
            cvar_results[i] = c
            qcvar_results[i] = qc
            var_results[i] = v
            qvar_results[i] = qv
            tvar_results[i] = t
            qtvar_results[i] = qt
            expectation_results[i] = e
            GC.enable(true)
            x = nothing
            p = nothing
            GC.gc()
        end
        results[dist] = DataFrame(n=experiments, cvar=cvar_results, qcvar=qcvar_results,
                         var=var_results, qvar=qvar_results, tvar=tvar_results,
                         qtvar=qtvar_results, expectation=expectation_results)
    end
    return results
end

"""
    benchmark_stocks(; trials = 5, window = 10)

Benchmark the risk-measure algorithms on distributions derived from real stock
data.

Reads `data/spy_data.csv`, and for each row builds a normal distribution from the
row's `Mean` and `Std` (rows with a missing `Mean` or `Std` are skipped). Each
algorithm (CVaR, qCVaR, VaR, qVaR, TVaR, qTVaR and the plain expectation) is then
timed over `trials` repetitions at risk level `α = 0.95`.

# Arguments
  - `trials`: number of repetitions per row (a warm-up run is performed first and
    discarded).
  - `window`: rolling window size carried with the data set.

# Returns
A `DataFrame` with one row per trial and columns `n`, `cvar`, `qcvar`, `var`,
`qvar`, `tvar`, `qtvar` and `expectation` holding the measured times in
milliseconds.
"""
function benchmark_stocks(; trials=5, window = 10)
    csv_path = joinpath(@__DIR__, "data", "spy_data.csv")

    df = CSV.File(csv_path) |> DataFrame
    println("Data loaded")
    println("Number of rows: ", size(df, 1))
    results =
        Vector{NamedTuple{(:n, :cvar, :qcvar, :var, :qvar, :tvar, :qtvar, :expectation),
                          Tuple{Int64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}}}()
    for i in ProgressBar(range(1, stop=size(df, 1)))
        GC.enable(false)

        # Get the data for the current window
        row = df[i, :]
        if ismissing(row[:Mean]) || ismissing(row[:Std])
            continue
        end
        μ = row[:Mean]
        σ = row[:Std]
        lower_return = μ - 4 * σ
        upper_return = μ + 4 * σ
        x = collect(range(lower_return, stop=upper_return, length=Int(1e4)))

        shuffle!(x)
        dist = Normal(μ, σ)

        p = pdf.(dist, x)
        p ./= sum(p)
        α = 0.95

        # Run the experiment 'trials' times to get a better estimate of the time
        run_one_experiment(x, p, α) # burn one for julia
        for j in 1:trials
            c, qc, v, qv, t, qt, e = run_one_experiment(x, p, α)
            push!(results, (n=i, cvar=c, qcvar=qc, var=v, qvar=qv, tvar=t, qtvar=qt, expectation=e))
        end
        GC.enable(true)
        x = nothing
        p = nothing
        GC.gc()
    end
    return DataFrame(results)
end


"""
Wrapper struct that keeps track of a dataset and its info for plotting.
"""
struct Plotter
    df::DataFrame
    slow_cols::Vector{String}        # which cols are slow
    fast_cols::Vector{String}        # which cols are fast
    col2Name::Dict{String,String}    # column name to display on graph
    col2Marker::Dict{String,Symbol}  # column name to marker
    col2Color::Dict{String,Symbol}   # column name to color
    CI::Dict{String,Tuple{Vector{Float64},Vector{Float64}}}  # (cis, means) per column
end

"""
    compute_cis_means(df, cols)

Compute the confidence intervals and means for each of `cols` over the unique
values of `n`.
"""
function compute_cis_means(df::DataFrame, cols::Vector{String})
    results = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()
    unique_sizes = unique(df.n)
    for col in cols
        cis = Float64[]
        means = Float64[]
        for size in unique_sizes
            df_size = df[df.n .== size, :]
            vals = df_size[!, col]
            ci = 1.96 * std(vals) / sqrt(length(vals))
            push!(cis, ci)
            push!(means, mean(vals))
        end
        results[col] = (cis, means)
    end
    return results
end

function Plotter(df, slow_cols, fast_cols, col2Name, col2Marker, col2Color)
    CI = compute_cis_means(df, vcat(slow_cols, fast_cols))
    Plotter(df, slow_cols, fast_cols, col2Name, col2Marker, col2Color, CI)
end

"""
    plot_means_and_cis!(plt, plotter, col)

Plot the mean line and a shaded confidence-interval ribbon for `col`.
"""
function plot_means_and_cis!(plt, plotter::Plotter, col::String)
    unique_sizes = unique(plotter.df.n)
    cis, means = plotter.CI[col]
    # Fast methods are drawn with a solid line, standard ones with a dashed line.
    linestyle = col in plotter.fast_cols ? :solid : :dash
    plot!(plt, unique_sizes, means;
        label = plotter.col2Name[col],
        marker = plotter.col2Marker[col],
        color = plotter.col2Color[col],
        linestyle = linestyle,
        ribbon = cis,
        fillalpha = 0.2)
    return plt
end

"""
    plot_all_slow_vs_fast(plotter)

Plot the comparison between all slow and fast methods on log-log axes and return
the plot object. Use `savefig` to write it to a file.
"""
function plot_all_slow_vs_fast(plotter::Plotter)
    plt = plot()
    for col in vcat(plotter.slow_cols, plotter.fast_cols)
        plot_means_and_cis!(plt, plotter, col)
    end
    plot!(plt; xscale = :log10, yscale = :log10,
        xlabel = "Size of Probability Space (n)", ylabel = "Time (ms)",
        legend = :topleft)
    return plt
end

"""
    plot_result(csvfile)

Loads a dataframe from a CSV file `csvfile` and plots it.
"""
function plot_result(csvfile)
    slow = ["cvar", "var", "tvar"]
    fast = ["qcvar", "qvar", "qtvar", "expectation"]
    col2Name = Dict("cvar" => "CVaR", "qcvar" => "QCVaR", "var" => "VaR",
        "qvar" => "QVaR", "tvar" => "TVaR", "qtvar" => "QTVaR", "expectation" => "E")
    col2Color = Dict("cvar" => :blue, "qcvar" => :blue, "var" => :green,
        "qvar" => :green, "tvar" => :purple, "qtvar" => :purple, "expectation" => :pink)
    # A method pair (standard/fast) shares both color and marker, e.g. CVaR/QCVaR.
    col2Marker = Dict("cvar" => :circle, "qcvar" => :circle, "var" => :rect,
        "qvar" => :rect, "tvar" => :diamond, "qtvar" => :diamond, "expectation" => :pentagon)

    # Columns: n, cvar, qcvar, var, qvar, tvar, qtvar, expectation
    df = CSV.File(csvfile) |> DataFrame
    plotter = Plotter(df, slow, fast, col2Name, col2Marker, col2Color)
    plot_all_slow_vs_fast(plotter)
end
