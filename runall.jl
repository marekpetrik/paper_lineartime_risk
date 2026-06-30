include("benchmark.jl")

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

println("Plotting!")

# Plot every benchmark CSV in the directory
for csvfile in filter(f -> endswith(f, ".csv"), readdir())
    println("Plotting $csvfile")
    plt = plot_result(csvfile)
    savefig(plt, replace(csvfile, ".csv" => ".pdf"))
end
