include("benchmark.jl")
include("risk_averse_value_iteration.jl")

###

if !isfile("benchmark_stocks.csv")
    df = benchmark_stocks()
    CSV.write("benchmark_stocks.csv", df)
end

if !isfile("benchmark_random_uniform.csv")
    results = benchmark_random()
    for (dist, df) in results
        CSV.write("benchmark_random_$dist.csv", df)
    end
end

if !isfile("benchmark_random_small.csv")
    results = benchmark_random(trials=10, start=1000, step=1000, stop=10000)
    CSV.write("benchmark_random_small.csv", results["sparse"])
end

### Generate the reinforcement learning results

if !isfile("benchmark_value_iteration.csv")
    df = benchmark_value_iteration()
    CSV.write("benchmark_value_iteration.csv", df)
end

println("Plotting!")

### Plot every benchmark CSV in the directory
for csvfile in filter(f -> endswith(f, ".csv"), readdir())
    println("Plotting $csvfile")
    csvfile == "benchmark_stocks.csv" && continue   # skip stocks
    xlabel = csvfile == "benchmark_value_iteration.csv" ? "Number of States" :
             "Size of Probability Space (n)"
    plot_and_save_results(csvfile, xlabel=xlabel)
end


### Generate Latex tables for the small datasets (saves them as tex)

println("Generating small table")
generate_tables("benchmark_random_small.csv")

println("Generating stocks table")
generate_tables("benchmark_stocks.csv")

println("Generating value iteration table")
generate_tables("benchmark_value_iteration.csv")
