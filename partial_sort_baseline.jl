# Partial sort baseline to compare our implentation against
# Fastest on uniform distributions
# Matches `RiskMeasures.VaR`: the VaR is x at the sorted position j, the smallest j whose
# cumulative mass S_j (mass of the j smallest atoms; largest if rev) exceeds α. x and p are
# left partially sorted together (with uniform p only x is reordered, since all entries of p
# are equal), and x[index] == value on return. Mass within atol of α counts as exceeding it.
# Partial sorting is not done when p is uniform
function PVaR!(x, p, α; rev::Bool=false, atol=0.0)
    n = length(x)
    T = float(eltype(x))
    α >= 1 && return (value=rev ? typemin(T) : typemax(T), index=-1)   # VaR_1 is infinite

    j = 0                         # VaR position, 0 = not found yet, -1 = mass ≤ α
    # Bounds on j, assuming sum(p) == 1. j atoms carry between j*pmin and j*pmax of the
    # mass, as do the n - j atoms above them, so
    #   S_j > α      needs  j > α / pmax  and  j > n - (1-α) / pmin,
    #   S_{j-1} ≤ α  needs  j ≤ α / pmin + 1  and  j ≤ n + 1 - (1-α) / pmax.
    # With uniform p all four agree and k == l is computed analytically.
    pmin, pmax = extrema(p)
    flr(t) = floor(Int, clamp(t, -1, n))   # floor that is safe for ±Inf
    k = max(1, flr(α / pmax) + 1)
    l = min(n, flr((n + 1) - ((1 - α) / pmax)))
    if pmin > 0   # pmin == 0 is common (sparse p), and 0 / 0 at α == 0 would be NaN
        k = max(k, flr(n - ((1 - α) / pmin)) + 1)
        l = min(l, flr(α / pmin) + 1)
    end
    k = clamp(k, 1, n)
    l = clamp(l, k, n)

    Tp = float(promote_type(eltype(p), typeof(α)))

    if pmin == pmax # Uniform p: reordering x alone keeps the distribution
        k = clamp(floor(Int, big(α - atol) * n) + 1, 1, n)   # big: n * α in Float64 can round onto an integer
        return (value=float(partialsort!(x, k; rev=rev)), index=k)
    end

    ix = collect(eachindex(x))    # permutation buffer
    w = 1                         # widening step
    while j == 0
        # ix[k:l] = indices of the k-th..l-th smallest (largest, if rev) entries of x, in
        # order, and ix[1:k-1] the indices of the entries before them, in any order.
        partialsortperm!(ix, x, k:l; rev=rev) # costs O(m log m) for m = l - k + 1, which while smaller than O(n log n) is still expensive for large n

        base = zero(Tp)
        for i in 1:(k-1)
            base += p[ix[i]]
        end

        if k > 1 && base > α - atol
            # j < k: possible only through rounding or p not summing to 1
            l = k - 1
            k = max(1, k - w)
        else
            cum = base
            for i in k:l
                cum += p[ix[i]]
                if cum > α - atol
                    j = i
                    break
                end
            end
            j == 0 && l == n && (j = -1)
            k = l + 1
            l = min(n, l + w)
        end
        w *= 2
    end

    permute!(x, ix)     # x is left partially sorted, as partialsort! would leave it
    permute!(p, ix)     # p follows the same permutation, so the pairing survives

    j == -1 && return (value=rev ? typemin(T) : typemax(T), index=-1)
    return (value=float(x[j]), index=j)
end
