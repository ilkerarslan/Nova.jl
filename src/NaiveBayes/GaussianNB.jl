using Statistics

import ...NovaML: AbstractModel, softmax

mutable struct GaussianNB <: AbstractModel
    # Learned parameters
    classes_::Vector
    class_count_::Vector{Float64}
    class_prior_::Vector{Float64}
    theta_::Matrix{Float64}          # per-class feature means (n_classes × n_features)
    var_::Matrix{Float64}            # per-class feature variances (n_classes × n_features)
    epsilon_::Float64                # absolute additive variance (smoothing)
    n_features_in_::Int
    fitted::Bool

    # Hyperparameters
    var_smoothing::Float64
    priors::Union{Nothing, Vector{Float64}}

    function GaussianNB(;
        var_smoothing::Real = 1e-9,
        priors::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
        if !isfinite(var_smoothing) || var_smoothing < 0
            throw(ArgumentError("`var_smoothing` must be a finite non-negative number, got $var_smoothing"))
        end
        p = priors === nothing ? nothing : Float64.(collect(priors))
        # Value-only validation of `priors` is data-independent, so it belongs in
        # the constructor alongside the `var_smoothing` check (fail fast, and stay
        # consistent with how the scalar hyperparameters are validated). The
        # data-dependent length check (priors-vs-class-count) necessarily stays in
        # fit, where the number of classes is known. Both checks are repeated in
        # fit so the `GridSearchCV` `setproperty!` mutation path — which bypasses
        # this constructor — is still guarded.
        if p !== nothing
            if any(!isfinite, p) || any(<(0), p)
                throw(ArgumentError("`priors` must contain only finite, non-negative values, got $p"))
            end
            if !isapprox(sum(p), 1.0; atol=1e-8)
                throw(ArgumentError("`priors` must sum to 1, got sum=$(sum(p))"))
            end
        end
        new(
            Any[], Float64[], Float64[],
            Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
            0.0, 0, false,
            Float64(var_smoothing), p
        )
    end
end

# ---- fit ----
function (m::GaussianNB)(X::AbstractMatrix, y::AbstractVector)
    if length(y) != size(X, 1)
        throw(DimensionMismatch("`y` length ($(length(y))) must match the number of rows in `X` ($(size(X, 1)))"))
    end
    # Reject empty / zero-feature training data up front. Without this, a 0-row X
    # leaves `maximum(Statistics.var(Xf, dims=1))` as NaN (silently producing a
    # NaN-fitted model), and a 0-feature X makes `maximum(...)` throw a cryptic
    # low-level "reducing over an empty collection" ArgumentError. A clear message
    # here matches scikit-learn's `check_array` (which rejects empty arrays at fit).
    if size(X, 1) == 0
        throw(ArgumentError("`X` has no samples; cannot fit GaussianNB on empty data"))
    end
    if size(X, 2) == 0
        throw(ArgumentError("`X` has no features; cannot fit GaussianNB"))
    end
    Xf = Float64.(X)
    # Reject non-finite feature values. A NaN/Inf entry propagates through
    # `mean`/`Statistics.var` into the learned `theta_`/`var_`, and from there into
    # every Gaussian log-likelihood: `argmax` over a NaN log-joint silently returns
    # an arbitrary class (verified: argmax([1.0, NaN, 2.0]) == 2). This mirrors
    # scikit-learn's `check_array` (force_all_finite=True), which rejects NaN/Inf at
    # fit. Done after the Float64 conversion so integer inputs (always finite) are a
    # cheap no-op.
    if any(!isfinite, Xf)
        throw(ArgumentError("`X` must contain only finite values for GaussianNB (no NaN/Inf)"))
    end
    classes = sort(unique(y))
    n_samples, n_features = size(Xf)
    k = length(classes)

    means      = zeros(Float64, k, n_features)
    variances  = zeros(Float64, k, n_features)
    class_count = zeros(Float64, k)

    for (idx, c) in enumerate(classes)
        rows = findall(==(c), y)
        Xc = Xf[rows, :]
        class_count[idx] = length(rows)
        means[idx, :]     = vec(mean(Xc, dims=1))
        variances[idx, :] = vec(Statistics.var(Xc, dims=1, corrected=false))
    end

    # sklearn-style absolute smoothing: var_smoothing * max(feature variance over all data)
    epsilon = m.var_smoothing * maximum(Statistics.var(Xf, dims=1, corrected=false))
    variances .+= epsilon

    # Degenerate-data guard: with var_smoothing=0 (allowed) a per-class constant
    # feature, or globally constant data, leaves a zero variance, which would yield
    # divide-by-zero / NaN posteriors. Floor every variance at a tiny positive value
    # so log/division stay finite. eps(Float64) ≈ 2.2e-16 is negligible for any
    # realistically-scaled feature, so this does not perturb normal fits.
    variances .= max.(variances, eps(Float64))

    if m.priors !== nothing
        if length(m.priors) != k
            throw(ArgumentError("`priors` length ($(length(m.priors))) must match number of classes ($k)"))
        end
        if !(all(m.priors .>= 0) && isapprox(sum(m.priors), 1.0; atol=1e-8))
            throw(ArgumentError("`priors` must be non-negative and sum to 1"))
        end
        class_prior = copy(m.priors)
    else
        class_prior = class_count ./ n_samples
    end

    m.classes_        = classes
    m.class_count_    = class_count
    m.class_prior_    = class_prior
    m.theta_          = means
    m.var_            = variances
    m.epsilon_        = epsilon
    m.n_features_in_  = n_features
    m.fitted          = true
    return m
end

# ---- per-class log joint for one sample ----
function _gnb_log_joint(m::GaussianNB, x::AbstractVector)
    k = length(m.classes_)
    lj = Vector{Float64}(undef, k)
    for c in 1:k
        v = view(m.var_, c, :)
        d = x .- view(m.theta_, c, :)
        ll = -0.5 * sum(log.(2π .* v)) - 0.5 * sum((d .^ 2) ./ v)
        lj[c] = log(m.class_prior_[c]) + ll
    end
    return lj
end

# ---- predict a single sample ----
function (m::GaussianNB)(x::AbstractVector)
    if !m.fitted
        throw(ErrorException("This GaussianNB instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    # Validate feature count. Without this, a length-1 `x` would broadcast against
    # the (n_classes × n_features) `theta_`/`var_` and silently produce a wrong
    # prediction instead of erroring; other mismatched lengths already throw a
    # low-level DimensionMismatch deep in `_gnb_log_joint`. A clear check here
    # matches scikit-learn, which validates n_features at predict.
    if length(x) != m.n_features_in_
        throw(DimensionMismatch("`x` has $(length(x)) features, but GaussianNB was fitted with $(m.n_features_in_)"))
    end
    # Reject non-finite inputs at predict too (matching scikit-learn's predict-time
    # `check_array`): a NaN/Inf feature makes `_gnb_log_joint` produce a NaN/Inf
    # log-joint, and `argmax` would then silently return a wrong label.
    if any(!isfinite, x)
        throw(ArgumentError("`x` must contain only finite values for GaussianNB (no NaN/Inf)"))
    end
    lj = _gnb_log_joint(m, x)
    return m.classes_[argmax(lj)]
end

# ---- predict a batch (labels, or class posteriors with type=:probs) ----
function (m::GaussianNB)(X::AbstractMatrix; type=nothing)
    if !m.fitted
        throw(ErrorException("This GaussianNB instance is not fitted yet. Call the model with training data before using it for predictions."))
    end
    # Validate feature count up front (see the single-sample predict above): a
    # 1-column `X` would otherwise broadcast row-by-row and silently mispredict.
    if size(X, 2) != m.n_features_in_
        throw(DimensionMismatch("`X` has $(size(X, 2)) features, but GaussianNB was fitted with $(m.n_features_in_)"))
    end
    # Reject non-finite inputs (see the single-sample predict above): a NaN/Inf entry
    # would silently mispredict via `argmax` over a NaN/Inf log-joint.
    if any(!isfinite, X)
        throw(ArgumentError("`X` must contain only finite values for GaussianNB (no NaN/Inf)"))
    end
    n = size(X, 1)
    k = length(m.classes_)
    lj = Matrix{Float64}(undef, n, k)
    for i in 1:n
        lj[i, :] = _gnb_log_joint(m, view(X, i, :))
    end
    if type == :probs
        return softmax(lj)
    else
        return [m.classes_[argmax(view(lj, i, :))] for i in 1:n]
    end
end

function Base.show(io::IO, m::GaussianNB)
    print(io, "GaussianNB(var_smoothing=$(m.var_smoothing), ",
        "priors=$(m.priors), fitted=$(m.fitted))")
end
