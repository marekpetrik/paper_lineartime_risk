module Benchmark

include("worstcase_l1.jl")

using Base.Threads
using Dates
using DataFrames
using CSV
using Random
using ProgressBars
using Distributions
using RiskMeasures
Random.seed!(1234)

"""
    run_one_experiment(n::Int, x, p, α)

Run one experiment with n samples and random variable *x* with probability distribution 
*p* and risk level *α* and return the results.

# Returns:
  NTuple{2, Float64}:
    slow_cvar_result: The CVaR time taken result.
    fast_cvar_result: The qCVaR time taken result.
"""
function run_one_experiment(x, p, α)
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
    tmpx = deepcopy(x)
    tmpp = deepcopy(p)
    start = time_ns()
    worstcase_l1(tmpx, tmpp, α)
    tvar_time = (time_ns() - start) * 1e-6
    tmpx = deepcopy(x)
    tmpp = deepcopy(p)
    start = time_ns()
    TVaR!(tmpx, tmpp, α)
    qtvar_time = (time_ns() - start) * 1e-6
    start = time_ns()
    expectation = sum(tmpx .* tmpp)
    expectation_time = (time_ns() - start) * 1e-6
    local δ = abs(slow_cvar_result - fast_cvar_result)
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
    benchmark_random(trials = 10, start = Int(1e6), step = Int(1e6), stop = Int(1e7))

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
function benchmark_random(trials = 10, start = Int(1e6), step = Int(1e6), stop = Int(1e7))
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
    benchmark_stocks(trials = 5, window = 10)

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
function benchmark_stocks(trials=5, window = 10)
    csv_path = joinpath(dirname(pathof(Benchmark)), "..", "data", "spy_data.csv")

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


export benchmark_random, benchmark_stocks

end # module Benchmark
