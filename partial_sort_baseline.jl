function partial_sort_uniform!(x, p, α)
  p_unif = p[1] # ASSUMES p IS UNIFORM
  if iszero(p_unif) 
    p_unif += 1e-4
  end
  k = Integer(floor(Int, (α / p_unif[1]) + 1)) 
  k = min(length(x), k) # For non-uniform case
  v = partialsort!(x, k)
  return (value=v, index=k)
end

function partial_sort_general!(x, p, α; rev::Bool=false, atol=0.0)
    n = length(x)

    # k atoms carry at most k * maximum(p) of the mass
    # start at k = α / maximum(p)
    pmax = maximum(p)
    guess = pmax > 0 ? α / pmax : Inf
    k = guess >= n ? n : max(1, ceil(Int, guess))

    ix = collect(eachindex(x))                  # working permutation buffer
    T = float(promote_type(eltype(p), typeof(α)))
    j = 0                                       # answer position, 0 = not found yet

    while true
        # ix[1:k] = indices of the k smallest (largest, if rev) entries of x, in order.
        # O(n) quickselect for the split + O(k log k) to order the prefix.
        partialsortperm!(ix, x, 1:k; rev=rev)

        # Recomputed from scratch each round: successive calls may break ties
        # between equal values differently, so an incremental sum is not safe.
        cum = zero(T)
        for i in 1:k
            cum += p[ix[i]]
            if cum >= α - atol
                j = i
                break
            end
        end

        (j > 0 || k == n) && break

        # Short of α: at least double k (keeps total work geometric) and
        # extrapolate further when the mass is coming in slowly.
        if cum > 0
            kf = k * (α / cum)
            knext = kf >= n ? n : ceil(Int, kf)
        else
            knext = 2k
        end
        k = min(n, max(2k, knext))
    end

    j == 0 && (j = n)   # α exceeds sum(p): saturate on the extreme atom

    permute!(x, ix)     # x is left partially sorted, as partialsort! would leave it
    permute!(p, ix)     # p follows the same permutation, so the pairing survives

    return (value = x[j], index = j)
end
