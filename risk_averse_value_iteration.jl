# implement risk averse nested cvar / var
using MDPs
using Base.Threads
using RiskMeasures
using CSV
using Plots
using Statistics
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

mdp = create_inventory_domain(400)


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

function benchmark_nestVi(mdp, T::Int, alphas::Vector{Float64}, γ::Float64)
  # --- CVaR (Fast) ---
  start = time_ns()
  res_cvar_fast = multi_alpha_nestVi(
    mdp, T, alphas, γ,
    (X, P, α) -> RiskMeasures.CVaR!(X, P, α, check_inputs=false, fast=true).value
  )
  fast_cvar_time = (time_ns() - start) * 1e-6

  # --- CVaR (Slow) ---
  start = time_ns()
  res_cvar_slow = multi_alpha_nestVi(
    mdp, T, alphas, γ,
    (X, P, α) -> RiskMeasures.CVaR!(X, P, α, check_inputs=false, fast=false).value
  )
  slow_cvar_time = (time_ns() - start) * 1e-6

  # --- VaR (Slow) ---
  start = time_ns()
  multi_alpha_nestVi(
    mdp, T, alphas, γ,
    (X, P, α) -> RiskMeasures.VaR!(X, P, α, check_inputs=false, fast=false).value
  )
  var_time = (time_ns() - start) * 1e-6

  # --- VaR (Fast) ---
  start = time_ns()
  multi_alpha_nestVi(
    mdp, T, alphas, γ,
    (X, P, α) -> RiskMeasures.VaR!(X, P, α, check_inputs=false, fast=true).value
  )
  qvar_time = (time_ns() - start) * 1e-6

  # --- TVaR (worstcase_l1) ---
  start = time_ns()
  multi_alpha_nestVi(
    mdp, T, alphas, γ,
    # Wraps the raw float output in a NamedTuple so .value extraction works
    (X, P, α) -> worstcase_l1(X, P, 2 * α)[2],
  )
  tvar_time = (time_ns() - start) * 1e-6

  # --- TVaR (choquet_ews) ---
  start = time_ns()
  multi_alpha_nestVi(
    mdp, T, alphas, γ,
    (X, P, α) -> choquet_ews(X, P, choquet_ews_tvar(α)).value,
  )
  qtvar_time = (time_ns() - start) * 1e-6

  # --- Expectation ---
  start = time_ns()
  multi_alpha_nestVi(
    mdp, T, alphas, γ,
    (X, P, α) -> dot(X, P),
  )
  expectation_time = (time_ns() - start) * 1e-6

  # --- Correctness Check ---
  # Compares the final value functions (at t=1) for slow vs fast CVaR
  δ = maximum(abs.(res_cvar_slow.value_function[1] .- res_cvar_fast.value_function[1]))
  if δ >= 1e-6
    println("Max diff: $δ")
    error("Results are not equal between slow and fast CVaR!")
  end

  return (
    slow_cvar_time=slow_cvar_time,
    fast_cvar_time=fast_cvar_time,
    var_time=var_time,
    qvar_time=qvar_time,
    tvar_time=tvar_time,
    qtvar_time=qtvar_time,
    expectation_time=expectation_time
  )
end

function benchmark_multiple_trials(mdp, T::Int, alphas::Vector{Float64}, γ::Float64; trials::Int=10)
  methods = [
    :slow_cvar_time, :fast_cvar_time,
    :var_time, :qvar_time,
    :tvar_time, :qtvar_time,
    :expectation_time
  ]

  # Preallocate a dictionary to store trial results
  results = Dict{Symbol,Vector{Float64}}(m => Float64[] for m in methods)

  println("Running $trials benchmark trials...")

  # Wrap the loop iterator in ProgressBar
  benchmark_nestVi(mdp, T, alphas, γ) # burn one for julia
  for i in ProgressBar(1:trials)
    trial_res = benchmark_nestVi(mdp, T, alphas, γ)
    for m in methods
      push!(results[m], getproperty(trial_res, m))
    end
  end

  # Compute mean and standard deviation
  stats = Dict{Symbol,Tuple{Float64,Float64}}()
  for m in methods
    stats[m] = (mean(results[m]), std(results[m]))
  end

  # Print a markdown table of the computed statistics to the console
  println("\n| Method | Mean Time (ms) | Std Dev (ms) |")
  println("| :--- | :--- | :--- |")
  println("| CVaR (Slow) | $(round(stats[:slow_cvar_time][1], digits=2)) | $(round(stats[:slow_cvar_time][2], digits=2)) |")
  println("| CVaR (Fast) | $(round(stats[:fast_cvar_time][1], digits=2)) | $(round(stats[:fast_cvar_time][2], digits=2)) |")
  println("| VaR (Slow) | $(round(stats[:var_time][1], digits=2)) | $(round(stats[:var_time][2], digits=2)) |")
  println("| VaR (Fast) | $(round(stats[:qvar_time][1], digits=2)) | $(round(stats[:qvar_time][2], digits=2)) |")
  println("| TVaR (Slow/L1) | $(round(stats[:tvar_time][1], digits=2)) | $(round(stats[:tvar_time][2], digits=2)) |")
  println("| TVaR (Fast/Choquet) | $(round(stats[:qtvar_time][1], digits=2)) | $(round(stats[:qtvar_time][2], digits=2)) |")
  println("| Expectation | $(round(stats[:expectation_time][1], digits=2)) | $(round(stats[:expectation_time][2], digits=2)) |")

  # Generate the Plot
  p = plot(
    title="Execution Time over $trials Trials",
    xlabel="Trial Number",
    ylabel="Time (ms)",
    legend=:outertopright,
    size=(800, 500),
    margin=5Plots.mm
  )

  x_axis = 1:trials

  # Add series to plot (Dotted for Slow, Solid for Fast)
  plot!(p, x_axis, results[:slow_cvar_time], label="CVaR (Slow)", ls=:dot, lw=2, color=:red)
  plot!(p, x_axis, results[:fast_cvar_time], label="CVaR (Fast)", ls=:solid, lw=2, color=:red)

  plot!(p, x_axis, results[:var_time], label="VaR (Slow)", ls=:dot, lw=2, color=:blue)
  plot!(p, x_axis, results[:qvar_time], label="VaR (Fast)", ls=:solid, lw=2, color=:blue)

  plot!(p, x_axis, results[:tvar_time], label="TVaR (Slow/L1)", ls=:dot, lw=2, color=:green)
  plot!(p, x_axis, results[:qtvar_time], label="TVaR (Fast/Choquet)", ls=:solid, lw=2, color=:green)

  plot!(p, x_axis, results[:expectation_time], label="Expectation", ls=:solid, lw=2, color=:black)

  return stats, p
end

# Execution
alphas = collect(0.0:0.1:1.0)
stats_summary, final_plot = benchmark_multiple_trials(mdp, 10, alphas, 0.95, trials=10)
savefig(final_plot, "benchmark_results.png")
