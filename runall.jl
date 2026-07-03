include("benchmark.jl")

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

if !isfile("benchmak_random_small.csv")
    results = benchmark_random(trials=10, start=1000,step=1000,stop=10000)
    CSV.write("benchmark_random_small.csv", results["sparse"])
end

println("Plotting!")

### Plot every benchmark CSV in the directory
for csvfile in filter(f -> endswith(f, ".csv"), readdir())
    println("Plotting $csvfile")
    csvfile == "benchmark_stocks.csv" && continue   # skip stocks
    plt = plot_result(csvfile)
    savefig(plt, replace(csvfile, ".csv" => ".pdf"))
end


### Generate Latex tables


csvfile = "benchmark_random_small.csv"
df = CSV.read(csvfile, DataFrame)

value_cols = setdiff(names(df), ["n"])
value_cols = ["expectation", "var", "qvar", "cvar", "qcvar", "tvar", "qtvar"]
grouped = groupby(df, :n)

df_mean = combine(grouped, [col => (x -> @sprintf("%.2f", mean(x))) => "$(col)" for col ∈ value_cols]...)

df_conf = combine(grouped, [col => (x -> @sprintf("%.2f", std(x) * 1.96 / sqrt(length(x)))) =>
    "$(col)" for col ∈ value_cols]...)

write("small_mean.tex",
      latexify(df_mean; env = :table, booktabs = true, latex = false, adjustment = :r,
               head = names(df)) )

write("small_std.tex", 
      latexify(df_conf; env = :table, booktabs = true, latex = false, adjustment = :r,
               head = names(df)) )


### Write stocks results
