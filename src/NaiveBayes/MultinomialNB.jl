using SparseArrays

import ...NovaML: AbstractModel, softmax

mutable struct MultinomialNB <: AbstractModel
    # Learned parameters
    classes_::Vector
    class_count_::Vector{Float64}
    feature_count_::Matrix{Float64}        # per-class summed feature counts (n_classes × n_features)
    class_log_prior_::Vector{Float64}      # log P(class) (length n_classes)
    feature_log_prob_::Matrix{Float64}     # log P(feature | class) (n_classes × n_features)
    n_features_in_::Int
    fitted::Bool

    # Hyperparameters
    alpha::Float64
    fit_prior::Bool
    class_prior::Union{Nothing, Vector{Float64}}

    function MultinomialNB(;
        alpha::Real = 1.0,
        fit_prior::Bool = true,
        class_prior::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
        if !isfinite(alpha) || alpha < 0
            throw(ArgumentError("`alpha` must be a finite non-negative number, got $alpha"))
        end
        cp = class_prior === nothing ? nothing : Float64.(collect(class_prior))
        # Value-only validation of `class_prior` is data-independent, so it belongs
        # in the constructor alongside the `alpha` check (fail fast, consistent with
        # the scalar hyperparameter). Note `class_prior` must be *strictly positive*
        # (not just non-negative): the fit takes `log.(class_prior)`, and a zero
        # entry would yield `-Inf`. The data-dependent length check stays in fit,
        # where the class count is known. Both checks are repeated in fit so the
        # `GridSearchCV` `setproperty!` mutation path — which bypasses this
        # constructor — is still guarded.
        if cp !== nothing
            if any(!isfinite, cp) || any(<=(0), cp)
                throw(ArgumentError("`class_prior` must contain only finite, positive values, got $cp"))
            end
            if !isapprox(sum(cp), 1.0; atol=1e-8)
                throw(ArgumentError("`class_prior` must sum to 1, got sum=$(sum(cp))"))
            end
        end
        new(
            Any[], Float64[],
            Matrix{Float64}(undef, 0, 0), Float64[], Matrix{Float64}(undef, 0, 0),
            0, false,
            Float64(alpha), fit_prior, cp
        )
    end
end

# ---- fit ----
function (m::MultinomialNB)(X::AbstractMatrix, y::AbstractVector)
    if length(y) != size(X, 1)
        throw(DimensionMismatch("`y` length ($(length(y))) must match the number of rows in `X` ($(size(X, 1)))"))
    end
    # Reject empty / zero-feature training data up front. A 0-row X yields zero
    # classes (k == 0) and a degenerate "fitted" model whose predict path errors
    # cryptically; a 0-feature X produces empty per-class log-probabilities. A clear
    # message here matches scikit-learn's `check_array` (which rejects empty arrays).
    if size(X, 1) == 0
        throw(ArgumentError("`X` has no samples; cannot fit MultinomialNB on empty data"))
    end
    if size(X, 2) == 0
        throw(ArgumentError("`X` has no features; cannot fit MultinomialNB"))
    end
    # Multinomial NB is only defined for non-negative count/frequency features.
    # `any(<(0), X)` is sparse-compatible (it scans stored entries) and rejecting
    # here avoids `log` of a negative smoothed count downstream (DomainError/NaN).
    if any(<(0), X)
        throw(ArgumentError("`X` must contain only non-negative values for MultinomialNB"))
    end
    # Reject non-finite feature values. The non-negativity check above is not
    # sufficient: `NaN < 0` is false and `Inf >= 0`, so NaN/Inf counts slip past it
    # and poison the model — a NaN summand makes `feature_count_` NaN, and an Inf
    # count makes `smoothed_cc` Inf so `log(Inf) - log(Inf) == NaN` (verified) in
    # `feature_log_prob_`. `any(!isfinite, X)` is sparse-compatible: `isfinite(0)`
    # is true, so structural zeros are ignored and only stored entries are scanned.
    # Mirrors scikit-learn's `check_array` (force_all_finite=True) at fit.
    if any(!isfinite, X)
        throw(ArgumentError("`X` must contain only finite values for MultinomialNB (no NaN/Inf)"))
    end
    classes = sort(unique(y))
    n_samples, n_features = size(X)
    k = length(classes)

    feature_count = zeros(Float64, k, n_features)
    class_count   = zeros(Float64, k)

    for (idx, c) in enumerate(classes)
        rows = findall(==(c), y)
        class_count[idx] = length(rows)
        feature_count[idx, :] = vec(sum(X[rows, :], dims=1))
    end

    # Laplace / Lidstone smoothing. `alpha=0` is accepted but clipped to a tiny
    # floor (matching scikit-learn's _ALPHA_MIN = 1e-10): a true zero would make
    # log P(feature|class) = -Inf for any unseen feature, and the predict dot
    # product 0 * -Inf = NaN. Clipping keeps every log-probability finite.
    alpha = max(m.alpha, 1e-10)
    smoothed_fc = feature_count .+ alpha              # k × n_features
    smoothed_cc = sum(smoothed_fc, dims=2)            # k × 1
    feature_log_prob = log.(smoothed_fc) .- log.(smoothed_cc)

    if m.class_prior !== nothing
        if length(m.class_prior) != k
            throw(ArgumentError("`class_prior` length ($(length(m.class_prior))) must match number of classes ($k)"))
        end
        if !(all(m.class_prior .> 0) && isapprox(sum(m.class_prior), 1.0; atol=1e-8))
            throw(ArgumentError("`class_prior` must be positive and sum to 1"))
        end
        class_log_prior = log.(m.class_prior)
    elseif m.fit_prior
        class_log_prior = log.(class_count) .- log(sum(class_count))
    else
        class_log_prior = fill(-log(k), k)
    end

    m.classes_           = classes
    m.class_count_       = class_count
    m.feature_count_     = feature_count
    m.feature_log_prob_  = feature_log_prob
    m.class_log_prior_   = vec(class_log_prior)
    m.n_features_in_     = n_features
    m.fitted             = true
    return m
end

# ---- predict a single sample ----
function (m::MultinomialNB)(x::AbstractVector)
    if !m.fitted
        throw(ErrorException("This MultinomialNB instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    # Reject non-finite inputs (matching scikit-learn's predict-time `check_array`).
    # Unlike negative values (which yield a finite, if meaningless, linear score and
    # are therefore left fit-only), a NaN/Inf entry makes `jll` non-finite and
    # `argmax` silently returns a wrong label.
    if any(!isfinite, x)
        throw(ArgumentError("`x` must contain only finite values for MultinomialNB (no NaN/Inf)"))
    end
    jll = m.feature_log_prob_ * x .+ m.class_log_prior_
    return m.classes_[argmax(jll)]
end

# ---- predict a batch (labels, or class posteriors with type=:probs) ----
function (m::MultinomialNB)(X::AbstractMatrix; type=nothing)
    if !m.fitted
        throw(ErrorException("This MultinomialNB instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    # Reject non-finite inputs (see the single-sample predict above): a NaN/Inf entry
    # makes `jll` non-finite and `argmax` silently mispredicts. `any(!isfinite, X)` is
    # sparse-compatible (structural zeros are finite, so only stored entries scan).
    if any(!isfinite, X)
        throw(ArgumentError("`X` must contain only finite values for MultinomialNB (no NaN/Inf)"))
    end
    jll = Matrix(X * m.feature_log_prob_') .+ m.class_log_prior_'   # n × k
    if type == :probs
        return softmax(jll)
    else
        n = size(X, 1)
        return [m.classes_[argmax(view(jll, i, :))] for i in 1:n]
    end
end

function Base.show(io::IO, m::MultinomialNB)
    print(io, "MultinomialNB(alpha=$(m.alpha), fit_prior=$(m.fit_prior), ",
        "class_prior=$(m.class_prior), fitted=$(m.fitted))")
end
