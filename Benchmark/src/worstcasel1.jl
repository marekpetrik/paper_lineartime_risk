
"""
Computes the worst case distribution with a bounded total deviation `ξ`
from the underlying probability distribution `p̄` for the random variable `z`.

Efficiently computes the solution of:
min_p   p^T * z
s.t.    || p - p̄ ||_1  ≤ ξ
        1^T p = 1
        p ≥ 0

Notes
-----
This implementation works in O(n log n) time because of the sort. Using
quickselect to choose the right quantile would work in O(n) time.

This function does not check whether the provided probability distribution sums
to 1.

Returns
-------
Optimal solution `p` and the objective value
"""
function worstcase_l1(z::Vector{<:Real}, p̄::Vector{<:Real}, ξ::Real)
    (maximum(p̄) ≤ 1 + 1e-9 && minimum(p̄) ≥ -1e-9)  ||
        "values must be between 0 and 1"
    ξ ≥ zero(ξ)|| "ξ must be nonnegative"
    (length(z) > 0 && length(z) == length(p̄)) ||
            "z's values needs to be same length as p̄'s values"
    
    ξ = clamp(ξ, 0, 2)
    size = length(z)
    sorted_ind = sortperm(z)

    out = copy(p̄)       #duplicate it
    k = sorted_ind[1]   #index begins at 1

    ϵ = min(ξ / 2, 1 - p̄[k])
    out[k] += ϵ
    i = size

    while ϵ > 0 && i > 0
        k = sorted_ind[i]
        i -= 1
        difference = min(ϵ, out[k])
        out[k] -= difference
        ϵ -= difference
    end

    return out, out'*z
end
