# Benchmark of risk-averse (nested) value iteration on an inventory MDP.
#
# This is a plain script (not a package). `include` it to get
# `benchmark_value_iteration` in your session; it is run from `runall.jl`.
# See README.md for full usage.

using MDPs
using Base.Threads
using RiskMeasures
using DataFrames
using ProgressBars
using LinearAlgebra
include(joinpath(@__DIR__, "worstcasel1.jl"))

function create_inventory_domain(target_state_count::Int)
  # 1. Calculate limits based on the requested state count
  # Assuming standard inventory states: max_inventory + max_backlog + 1 = total_states
  max_inv = floor(Int, target_state_count * 0.8)
  max_backlog = target_state_count - max_inv - 1

  # Scale maximum order relative to the state space
  max_ord = floor(Int, target_state_count * 0.5)

  # Fallbacks for edge cases if a very small state count is passed
  max_inv = max(max_inv, 1)
  max_backlog = max(max_backlog, 0)
  max_ord = max(max_ord, 1)

  # 2. Scale the demand profile relative to the state space
  max_dem = max(floor(Int, target_state_count * 0.5), 10)
  demand_vals = collect(0:max_dem)

  # Center the bell curve in the middle of the demand range
  center_dem = max_dem / 2.0
  raw_probs = [exp(-0.05 * (x - center_dem)^2) for x in demand_vals]
  demand_probs = raw_probs ./ sum(raw_probs)

  demand = MDPs.Domains.Inventory.Demand(demand_vals, demand_probs)

  # 3. Static Costs
  costs = MDPs.Domains.Inventory.Costs(12.0, 75.0, 3.5, 15.0)

  # 4. Construct the domain
  limits = MDPs.Domains.Inventory.Limits(max_inv, max_backlog, max_ord)
  params = MDPs.Domains.Inventory.Parameters(demand, costs, 10, limits)

  return MDPs.Domains.Inventory.Model(params)
end

# Assuming the risk measure function signature is: ρ(X, P, α) 
# where X is the vector of values, P is the vector of probabilities, and α is the risk parameter.
function q_value_alpha(mdp, s, a, v_next::Matrix{Float64}, α::Float64, α_idx::Int, γ::Float64, ρ::Function)
  X = Float64[]
  P = Float64[]

  for (s′, p, r) in transition(mdp, s, a)
    push!(X, r + γ * v_next[s′, α_idx])
    push!(P, p)
  end

  return ρ(X, P, α)
end

function multi_alpha_greedy(mdp, s::Int, v_next::Matrix{Float64}, alphas::Vector{Float64}, γ::Float64, ρ::Function)
  acts = actions(mdp, s)
  n_alphas = length(alphas)

  best_policy = Vector{Int}(undef, n_alphas)
  best_qvals = Vector{Float64}(undef, n_alphas)

  for (α_idx, α) in enumerate(alphas)
    max_q = -Inf
    best_a = first(acts)
    for a in acts
      q_val = q_value_alpha(mdp, s, a, v_next, α, α_idx, γ, ρ)
      if q_val > max_q
        max_q = q_val
        best_a = a
      end
    end
    best_policy[α_idx] = best_a
    best_qvals[α_idx] = max_q
  end

  return (policy=best_policy, qvalue=best_qvals)
end

function multi_alpha_nestVi(mdp, T::Int, alphas::Vector{Float64}, γ::Float64, ρ::Function)
  n_S = state_count(mdp)
  n_alphas = length(alphas)

  # v[t][s, α_idx] -> Matrix of |S| x |alphas| for each time step
  v = Vector{Matrix{Float64}}(undef, T + 1)

  # π[t][s][α_idx] -> Best action for each state and alpha
  π = Vector{Vector{Vector{Int}}}(undef, T)

  # Initialize terminal value v_{T+1}(s, α) = 0.0
  v[end] = zeros(Float64, n_S, n_alphas)

  for t in T:-1:1
    v[t] = zeros(Float64, n_S, n_alphas)
    π[t] = [zeros(Int, n_alphas) for s in 1:n_S]

    # Parallelize across states
    Threads.@threads for s in 1:n_S
      bg = multi_alpha_greedy(mdp, s, v[t+1], alphas, γ, ρ)

      # Store the computed q-values and policy for all alphas for state s
      v[t][s, :] = bg.qvalue
      π[t][s] = bg.policy
    end
  end

  return (value_function=v, policy=π, alphas=alphas)
end

"""
    benchmark_nestVi(mdp, T, alphas, γ)

Run nested value iteration on `mdp` once with each risk measure and return a
NamedTuple of the measured times (in milliseconds), keyed by the same column
names as `run_one_experiment` in `benchmark.jl`.
"""
function benchmark_nestVi(mdp, T::Int, alphas::Vector{Float64}, γ::Float64)
  risk_measures = (
    qcvar=(X, P, α) -> RiskMeasures.CVaR!(X, P, α, check_inputs=false, fast=true).value,
    cvar=(X, P, α) -> RiskMeasures.CVaR!(X, P, α, check_inputs=false, fast=false).value,
    var=(X, P, α) -> RiskMeasures.VaR!(X, P, α, check_inputs=false, fast=false).value,
    qvar=(X, P, α) -> RiskMeasures.VaR!(X, P, α, check_inputs=false, fast=true).value,
    tvar=(X, P, α) -> worstcase_l1(X, P, 2 * α)[2],
    qtvar=(X, P, α) -> choquet_ews(X, P, choquet_ews_tvar(α)).value,
    expectation=(X, P, α) -> dot(X, P),
  )

  times = Dict{Symbol,Float64}()
  values = Dict{Symbol,Matrix{Float64}}()
  for (name, ρ) in pairs(risk_measures)
    start = time_ns()
    res = multi_alpha_nestVi(mdp, T, alphas, γ, ρ)
    times[name] = (time_ns() - start) * 1e-6
    values[name] = res.value_function[1]
  end

  # --- Correctness Check ---
  # Compares the final value functions (at t=1) for slow vs fast CVaR
  δ = maximum(abs.(values[:cvar] .- values[:qcvar]))
  if δ >= 1e-6
    println("Max diff: $δ")
    error("Results are not equal between slow and fast CVaR!")
  end

  return (; (name => times[name] for name in keys(risk_measures))...)
end

"""
    benchmark_value_iteration(; trials = 10, states = [50, 100, 200, 400], T = 10,
                                alphas = collect(0.0:0.1:1.0), γ = 0.95)

Benchmark risk-averse nested value iteration on inventory MDPs.

For each state count in `states`, builds an inventory domain with
`create_inventory_domain` and times `T` steps of nested value iteration for all
risk levels in `alphas` with each risk measure (CVaR, qCVaR, VaR, qVaR, TVaR,
qTVaR and the plain expectation), repeating each `trials` times (a warm-up run is
performed first and discarded).

# Returns
A `DataFrame` with one row per trial and columns `n` (number of states), `cvar`,
`qcvar`, `var`, `qvar`, `tvar`, `qtvar` and `expectation` holding the measured
times in milliseconds.
"""
function benchmark_value_iteration(; trials=10, states=[50, 100, 200, 400], T=10,
                                   alphas=collect(0.0:0.1:1.0), γ=0.95)
  results = DataFrame()
  for n in states
    println("Running value iteration with $n states")
    mdp = create_inventory_domain(n)
    benchmark_nestVi(mdp, T, alphas, γ) # burn one for julia
    for _ in ProgressBar(1:trials)
      push!(results, merge((n=n,), benchmark_nestVi(mdp, T, alphas, γ)))
    end
  end
  return select(results, :n, :cvar, :qcvar, :var, :qvar, :tvar, :qtvar, :expectation)
end
