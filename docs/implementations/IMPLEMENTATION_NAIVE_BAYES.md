# Implementation Plan: NaiveBayes (GaussianNB & MultinomialNB)

## Overview

Add a new `NaiveBayes` submodule to NovaML.jl implementing two scikit-learn-style
estimators:

- **`GaussianNB`** — Naive Bayes for continuous features. Learns per-class feature
  means and variances plus class priors; computes class posteriors with Gaussian
  log-likelihoods in log space for numerical stability.
- **`MultinomialNB`** — Naive Bayes for count/frequency features (the natural
  partner for `CountVectorizer` / `TfidfVectorizer` text features). Learns per-class
  feature log-probabilities with Laplace/Lidstone (`alpha`) smoothing and either
  fitted, user-supplied, or uniform class priors.

Both follow the **functor convention** documented in `CLAUDE.md`:

- `nb(X::AbstractMatrix, y::AbstractVector)` — fits in place, sets `fitted = true`
  and all learned-attribute fields, returns `nb`.
- `nb(x::AbstractVector)` — predicts the label for a single sample.
- `nb(X::AbstractMatrix; type=nothing)` — predicts labels for a batch; with
  `type=:probs` returns the class-posterior matrix (rows sum to 1).
- Pre-fit prediction throws `ErrorException` with the standard
  *"This … instance is not fitted yet."* message.

Both subtype `AbstractModel` and import shared symbols with the nested relative
path `import ...NovaML: AbstractModel, softmax` (three dots — the algorithm files
sit two levels below the top module: `NovaML` → `NaiveBayes` → `GaussianNB.jl`).

### Files to create

- `src/NaiveBayes/NaiveBayes.jl` — submodule definition (`include` + `export`)
- `src/NaiveBayes/GaussianNB.jl`
- `src/NaiveBayes/MultinomialNB.jl`

### Files to modify

- `src/NovaML.jl` — `include` the submodule and add `NaiveBayes` to the `export` line
- `test/runtests.jl` — add `@testset` blocks for both models
- `README.md` — add a `### NaiveBayes` subsection under "Main Components"

### Design notes / pitfalls observed in the codebase

- **Matrix dispatch precedence.** `src/_types.jl` defines the fallback
  `(m::AbstractModel)(X::AbstractMatrix) = [m(x) for x in eachrow(X)]`. Because
  `GaussianNB`/`MultinomialNB` are *more specific* than `AbstractModel`, defining
  `(m::GaussianNB)(X::AbstractMatrix; type=nothing)` overrides the fallback for
  both the no-kwarg and the `type=:probs` calls — this is exactly the
  `LogisticRegression` pattern (`src/LinearModel/LogisticRegression.jl:125`). Keep
  the three distinct signatures `(X, y)` / `(x::AbstractVector)` /
  `(X::AbstractMatrix; type)` so fit, single-predict, and batch-predict never
  collide.
- **`softmax` operates row-wise.** `softmax` in `src/_methods.jl:23` subtracts
  `maximum(X, dims=2)` and normalizes with `sum(..., dims=2)` — it expects an
  `n_samples × n_classes` matrix. Build the full log-joint matrix and pass it
  straight to `softmax`; do not call it on a bare vector.
- **`CountVectorizer` refits on every raw-document call.** `cv(docs)` always
  rebuilds the vocabulary (`src/FeatureExtraction/CountVectorizer.jl:70`); there is
  no transform-only path for raw documents. The `MultinomialNB` text test therefore
  fits the vectorizer once on the training documents and evaluates **in-sample**
  (predict on the same `X`) — do not call `cv()` a second time on held-out docs
  expecting the same vocabulary.
- **Sparse inputs.** `CountVectorizer` returns a `SparseMatrixCSC`. The
  `MultinomialNB` fit/predict code must work on sparse `X`: `X[rows, :]`,
  `sum(X[rows,:], dims=1)`, and `X * feature_log_prob_'` all work on sparse
  matrices; wrap the predict product in `Matrix(...)` so the result is dense before
  broadcasting the prior. `using SparseArrays` is added to the file for safety.
- **`var` name.** `Statistics.var` is qualified explicitly in `GaussianNB` to avoid
  any ambiguity with the `var_` field/locals.
- **Input validation.** Both fits assert `length(y) == size(X, 1)` (raising
  `DimensionMismatch`) so a row/label mismatch fails loudly instead of silently
  mistraining or throwing a low-level `BoundsError`. Both fits also reject empty
  training data — `size(X, 1) == 0` (no samples) and `size(X, 2) == 0` (no
  features) raise `ArgumentError`. Without this guard a 0-row `X` leaves
  `GaussianNB`'s `maximum(var(...))` as `NaN` (silently producing a NaN-fitted
  model) and gives `MultinomialNB` zero classes (a degenerate "fitted" state),
  while a 0-feature `X` makes `GaussianNB`'s `maximum(...)` throw a cryptic
  "reducing over an empty collection" error. This mirrors scikit-learn's
  `check_array`, which rejects empty arrays at fit. `MultinomialNB` additionally
  rejects negative feature values (`any(<(0), X)`, sparse-compatible) since
  Multinomial NB is only defined for non-negative counts and a negative smoothed
  count would feed a `DomainError` into `log`. This negative-value check is
  *deliberately fit-only*, matching scikit-learn (whose `MultinomialNB` calls
  `check_non_negative` in `fit` but not in the predict-time `_check_X`). At predict
  the joint-log-likelihood is just the linear form `feature_log_prob_ * x .+
  class_log_prior_`: negative inputs yield a finite (if semantically meaningless)
  score, never a `NaN`/`DomainError`, so there is no numerical hazard to guard and
  a per-call non-negativity scan would add cost without matching sklearn.
- **Non-finite feature values (`X`).** Distinct from the negative-value check, both
  models reject `NaN`/`Inf` feature values at **both fit and predict** via
  `any(!isfinite, X)` (sparse-compatible: `isfinite(0)` is true, so structural
  zeros are skipped and only stored entries scan). This is necessary because every
  feature-value guard already in the plan misses non-finite inputs: `size(X,·)==0`,
  `length(y)!=size(X,1)`, the feature-count check, and `any(<(0), X)` all pass a
  `NaN`/`Inf` through (`NaN < 0` is `false`, `Inf >= 0`). The hazard is silent, not
  loud: in `GaussianNB` a non-finite entry poisons `mean`/`Statistics.var` → the
  learned `theta_`/`var_` → the log-joint; in `MultinomialNB` a `NaN` count poisons
  `feature_count_` and an `Inf` count makes `log(Inf) - log(Inf) == NaN` in
  `feature_log_prob_`. In every case `argmax` over a `NaN` log-joint returns an
  arbitrary class (verified: `argmax([1.0, NaN, 2.0]) == 2`) rather than erroring,
  so the prediction is silently wrong. Unlike the deliberately fit-only
  non-negativity check (whose rejected inputs would otherwise still produce a
  *finite* score), the finite check belongs at predict too: it both prevents the
  silent mispredict and mirrors scikit-learn's `check_array`, which applies
  `force_all_finite=True` at fit **and** predict. Both use `ArgumentError`.
- **Predict-time feature-count validation (`GaussianNB`).** Both predict paths
  assert the input feature count matches `n_features_in_` (`length(x)` for the
  single-sample call, `size(X, 2)` for the batch call), raising
  `DimensionMismatch`. Without this, a length-1 vector (or 1-column matrix row)
  would *broadcast* against the `(n_classes × n_features)` `theta_`/`var_` inside
  `_gnb_log_joint` (`x .- view(m.theta_, c, :)`) and silently produce a wrong
  prediction/posterior; other mismatched lengths already throw a low-level
  `DimensionMismatch` from the broadcast, so the explicit check mainly closes the
  silent length-1 case and yields a clearer message, mirroring scikit-learn's
  predict-time `n_features_in_` check. `MultinomialNB` needs no such guard: its
  predict uses matrix multiplication (`feature_log_prob_ * x`,
  `X * feature_log_prob_'`), which already rejects any feature-count mismatch with
  `DimensionMismatch`.
- **`alpha=0` / zero-variance degeneracy.** `MultinomialNB` accepts `alpha=0` but
  clips it to `1e-10` in fit (matching scikit-learn's `_ALPHA_MIN`); a literal
  zero produces `-Inf` log-probabilities and `0 * -Inf = NaN` at predict time.
  `GaussianNB` floors every learned variance at `eps(Float64)` after smoothing so
  constant-feature / `var_smoothing=0` data cannot drive a divide-by-zero `NaN`
  posterior. Both guards are negligible for normally-scaled data.
- **Non-finite hyperparameters.** A bare `var_smoothing < 0` / `alpha < 0` check is
  *insufficient*: in Julia both `NaN < 0` and `Inf < 0` evaluate to `false`, so a
  `NaN`/`Inf` would slip through and poison the fit. `var_smoothing=NaN` makes
  `epsilon = NaN * maximum(...)` NaN and the `max.(variances, eps)` floor does not
  rescue it (`max(NaN, eps) == NaN`), yielding a NaN-fitted model; `var_smoothing=Inf`
  blows the variances (and hence posteriors) up to `Inf`/`NaN`. Likewise
  `alpha=NaN`/`Inf` poisons `MultinomialNB`'s `feature_log_prob_`
  (`max(NaN,1e-10)==NaN`; `log(Inf)-log(Inf)==NaN`). Both constructors therefore
  validate with `!isfinite(x) || x < 0`, rejecting negative *and* non-finite values
  with an `ArgumentError`, mirroring scikit-learn's parameter validation.
- **Prior-vector validation is split by data-dependence.** The *value*-only checks
  on `priors` / `class_prior` (finite, non-negative for `GaussianNB` /
  strictly-positive for `MultinomialNB`, and sum-to-1) are data-independent, so —
  like the scalar `var_smoothing` / `alpha` checks above — they run in the
  **constructor** and fail fast on an obviously-invalid vector. Only the *length*
  check (priors-vs-class-count) is deferred to **fit**, where the number of classes
  is first known. To stay robust against the `GridSearchCV` `setproperty!` path
  (which mutates the struct directly, bypassing the constructor), fit *also*
  re-validates the full set (length **and** value). All of these raise
  `ArgumentError`. Note `MultinomialNB`'s `class_prior` must be strictly positive
  (`<=(0)` rejected), because fit takes `log.(class_prior)` and a zero entry would
  give `-Inf`; `GaussianNB`'s `priors` are only required non-negative.
- **Julia 1.6.7 floor.** No syntax newer than 1.6 (no `@something`-only-1.7 macros,
  no `Returns`, etc.). `argmax`, `findall(==(c), y)`, broadcasting, and `2π` are all
  fine on 1.6.

---

## Section 1: Create `src/NaiveBayes/GaussianNB.jl`

Create the file `src/NaiveBayes/GaussianNB.jl` with the following complete contents:

```julia
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
```

### Verification for Section 1

Parse-level check (does not require the package to load):

```bash
cd /Users/ilker/Documents/NovaML.jl && julia -e 'Meta.parse("begin\n" * read("src/NaiveBayes/GaussianNB.jl", String) * "\nend"); println("GaussianNB parse OK")'
```

Expected: `GaussianNB parse OK`. (A bare `import ...NovaML` line only resolves
inside the package, so a standalone `include` is *not* expected to work — the parse
check confirms syntactic validity. End-to-end behavior is verified in Section 5.)

---

## Section 2: Create `src/NaiveBayes/MultinomialNB.jl`

Create the file `src/NaiveBayes/MultinomialNB.jl` with the following complete
contents:

```julia
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
```

### Verification for Section 2

```bash
cd /Users/ilker/Documents/NovaML.jl && julia -e 'Meta.parse("begin\n" * read("src/NaiveBayes/MultinomialNB.jl", String) * "\nend"); println("MultinomialNB parse OK")'
```

Expected: `MultinomialNB parse OK`.

---

## Section 3: Create the submodule file `src/NaiveBayes/NaiveBayes.jl`

Create `src/NaiveBayes/NaiveBayes.jl` with these exact contents (mirrors the
`src/Neighbors/Neighbors.jl` shape):

```julia
module NaiveBayes

include("GaussianNB.jl")
export GaussianNB

include("MultinomialNB.jl")
export MultinomialNB

end
```

### Verification for Section 3

```bash
cd /Users/ilker/Documents/NovaML.jl && julia -e 'Meta.parse(read("src/NaiveBayes/NaiveBayes.jl", String)); println("NaiveBayes module parse OK")'
```

Expected: `NaiveBayes module parse OK`.

---

## Section 4: Wire the submodule into `src/NovaML.jl`

### Modify `src/NovaML.jl`

The current block (lines 23–30) reads:

```julia
include("Tree/Tree.jl")
include("Ensemble/Ensemble.jl")
include("LinearModel/LinearModel.jl")
include("MultiClass/MultiClass.jl")
include("Neighbors/Neighbors.jl")
include("SVM/SVM.jl")

export Tree, Ensemble, LinearModel, MultiClass, Neighbors, SVM
```

Replace it with (adds the `NaiveBayes` include after `SVM` and appends
`NaiveBayes` to the export line):

```julia
include("Tree/Tree.jl")
include("Ensemble/Ensemble.jl")
include("LinearModel/LinearModel.jl")
include("MultiClass/MultiClass.jl")
include("Neighbors/Neighbors.jl")
include("SVM/SVM.jl")
include("NaiveBayes/NaiveBayes.jl")

export Tree, Ensemble, LinearModel, MultiClass, Neighbors, SVM, NaiveBayes
```

Make no other edits to `src/NovaML.jl`.

### Verification for Section 4

```bash
cd /Users/ilker/Documents/NovaML.jl && julia --project=. -e '
    using NovaML
    gnb = NovaML.NaiveBayes.GaussianNB()
    mnb = NovaML.NaiveBayes.MultinomialNB()
    println(gnb)
    println(mnb)
    println("Module wiring OK")
'
```

Expected output includes the `GaussianNB(...)` and `MultinomialNB(...)` show
strings (both `fitted=false`) and `Module wiring OK`. This also confirms the
package still loads cleanly (no regression to existing submodules).

---

## Section 5: Add tests to `test/runtests.jl`

### Modify `test/runtests.jl`

Insert two new `@testset` blocks **inside** the outer `@testset "NovaML.jl"`
block, immediately before its closing `end` (i.e. after the
`GradientBoostingRegressor` testset that currently ends on line 363, and before the
final `end` on line 365). Do not modify the existing testsets.

Add the following text between the end of the `GradientBoostingRegressor` testset
and the final `end`:

```julia
    @testset "GaussianNB" begin
        using NovaML.NaiveBayes: GaussianNB

        @testset "Constructor defaults" begin
            gnb = GaussianNB()
            @test gnb.var_smoothing == 1e-9
            @test gnb.priors === nothing
            @test gnb.fitted == false
            @test gnb.n_features_in_ == 0
            @test isempty(gnb.classes_)
        end

        @testset "Constructor custom params" begin
            gnb = GaussianNB(var_smoothing=1e-6, priors=[0.3, 0.7])
            @test gnb.var_smoothing == 1e-6
            @test gnb.priors == [0.3, 0.7]
        end

        @testset "Constructor validation" begin
            @test_throws ArgumentError GaussianNB(var_smoothing=-1.0)
            # NaN/Inf pass a bare `< 0` check (both compare false) but would
            # propagate into NaN/Inf learned variances and posteriors.
            @test_throws ArgumentError GaussianNB(var_smoothing=NaN)
            @test_throws ArgumentError GaussianNB(var_smoothing=Inf)
            # Value-only `priors` validation is data-independent, so it happens in
            # the constructor (fail fast), not just at fit. Length validation needs
            # the class count and stays in fit (see "Bad priors rejected at fit").
            @test_throws ArgumentError GaussianNB(priors=[0.3, 0.3])      # sums to 0.6, not 1
            @test_throws ArgumentError GaussianNB(priors=[-0.1, 1.1])     # negative entry
            @test_throws ArgumentError GaussianNB(priors=[NaN, 0.5])      # non-finite entry
            @test_throws ArgumentError GaussianNB(priors=[Inf, 0.5])      # non-finite entry
        end

        @testset "Predict not fitted" begin
            gnb = GaussianNB()
            @test_throws ErrorException gnb([1.0, 2.0])
            @test_throws ErrorException gnb([1.0 2.0; 3.0 4.0])
        end

        @testset "Fit sets learned attributes" begin
            X = [1.0 1.0; 1.2 0.8; 0.8 1.1; 10.0 10.0; 10.2 9.8; 9.8 10.1]
            y = [0, 0, 0, 1, 1, 1]
            gnb = GaussianNB()
            result = gnb(X, y)
            @test result === gnb
            @test gnb.fitted == true
            @test gnb.classes_ == [0, 1]
            @test gnb.n_features_in_ == 2
            @test size(gnb.theta_) == (2, 2)
            @test size(gnb.var_) == (2, 2)
            @test gnb.class_count_ == [3.0, 3.0]
            @test gnb.class_prior_ ≈ [0.5, 0.5]
            # class-0 mean of feature 1 = mean(1.0, 1.2, 0.8) = 1.0
            @test gnb.theta_[1, 1] ≈ 1.0
        end

        @testset "Predict — well-separated classes" begin
            X = [1.0 1.0; 1.2 0.8; 0.8 1.1; 10.0 10.0; 10.2 9.8; 9.8 10.1]
            y = [0, 0, 0, 1, 1, 1]
            gnb = GaussianNB()
            gnb(X, y)

            # In-sample predictions recover the labels on a separable problem
            @test gnb(X) == y
            # Single-sample predict
            @test gnb([1.0, 1.0]) == 0
            @test gnb([10.0, 10.0]) == 1
            # A clearly class-0 query
            @test gnb([0.9, 1.0]) == 0
        end

        @testset "Predict probabilities" begin
            X = [1.0 1.0; 1.2 0.8; 0.8 1.1; 10.0 10.0; 10.2 9.8; 9.8 10.1]
            y = [0, 0, 0, 1, 1, 1]
            gnb = GaussianNB()
            gnb(X, y)

            probs = gnb(X; type=:probs)
            @test size(probs) == (6, 2)
            @test all(probs .>= 0)
            @test all(isapprox.(sum(probs, dims=2), 1.0; atol=1e-8))
            # First sample is class 0 → higher posterior in column 1
            @test probs[1, 1] > probs[1, 2]
            # Last sample is class 1 → higher posterior in column 2
            @test probs[6, 2] > probs[6, 1]
        end

        @testset "Custom priors used" begin
            X = [1.0 1.0; 1.2 0.8; 0.8 1.1; 10.0 10.0; 10.2 9.8; 9.8 10.1]
            y = [0, 0, 0, 1, 1, 1]
            gnb = GaussianNB(priors=[0.4, 0.6])
            gnb(X, y)
            @test gnb.class_prior_ ≈ [0.4, 0.6]
        end

        @testset "Bad priors length rejected at fit" begin
            X = [1.0 1.0; 10.0 10.0]
            y = [0, 1]
            # Wrong length is the genuine fit-time rejection: [0.5, 0.3, 0.2] is a
            # valid prior vector by value (sums to 1, all positive, finite) so it
            # clears the constructor, then fails the length check in fit against the
            # 2 classes. (Value-only failures — sum≠1, negative, non-finite — are now
            # caught earlier, in the constructor; see "Constructor validation".)
            @test_throws ArgumentError GaussianNB(priors=[0.5, 0.3, 0.2])(X, y)
        end

        @testset "Mismatched X/y rows rejected" begin
            X = [1.0 1.0; 2.0 2.0; 3.0 3.0]
            @test_throws DimensionMismatch GaussianNB()(X, [0, 1])
        end

        @testset "Mismatched feature count at predict rejected" begin
            X = [1.0 1.0; 1.2 0.8; 0.8 1.1; 10.0 10.0; 10.2 9.8; 9.8 10.1]
            y = [0, 0, 0, 1, 1, 1]
            gnb = GaussianNB()
            gnb(X, y)
            # A length-1 vector would otherwise broadcast across the 2-feature
            # model and silently mispredict; too-many features must also be caught.
            @test_throws DimensionMismatch gnb([1.0])
            @test_throws DimensionMismatch gnb([1.0, 2.0, 3.0])
            # Batch predict with the wrong number of columns
            @test_throws DimensionMismatch gnb([1.0 2.0 3.0; 4.0 5.0 6.0])
            @test_throws DimensionMismatch gnb(reshape([1.0, 2.0], 2, 1))
        end

        @testset "Empty / zero-feature data rejected" begin
            # No samples (0-row X with matching empty y): would otherwise leave a
            # NaN epsilon and silently produce a NaN-fitted model.
            @test_throws ArgumentError GaussianNB()(Matrix{Float64}(undef, 0, 2), Int[])
            # No features (0-column X): would otherwise throw a cryptic
            # "reducing over an empty collection" error from `maximum`.
            @test_throws ArgumentError GaussianNB()(Matrix{Float64}(undef, 3, 0), [0, 1, 1])
        end

        @testset "Non-finite features rejected" begin
            # NaN/Inf pass the empty, row/label, feature-count and (for MNB)
            # non-negativity guards, but silently poison mean/var → posteriors;
            # argmax over a NaN log-joint returns an arbitrary class. Reject at fit
            # and at predict, mirroring scikit-learn's check_array.
            Xnan = [1.0 1.0; NaN 0.8; 0.8 1.1; 10.0 10.0; 10.2 9.8; 9.8 10.1]
            Xinf = [1.0 1.0; Inf 0.8; 0.8 1.1; 10.0 10.0; 10.2 9.8; 9.8 10.1]
            y = [0, 0, 0, 1, 1, 1]
            @test_throws ArgumentError GaussianNB()(Xnan, y)
            @test_throws ArgumentError GaussianNB()(Xinf, y)

            # Fit on clean data, then reject non-finite predict inputs.
            Xgood = [1.0 1.0; 1.2 0.8; 0.8 1.1; 10.0 10.0; 10.2 9.8; 9.8 10.1]
            gnb = GaussianNB()
            gnb(Xgood, y)
            @test_throws ArgumentError gnb([NaN, 1.0])
            @test_throws ArgumentError gnb([Inf, 1.0])
            @test_throws ArgumentError gnb([1.0 NaN; 2.0 3.0])
            @test_throws ArgumentError gnb([1.0 2.0; Inf 3.0])
        end

        @testset "Degenerate constant data — no NaN" begin
            # Globally constant feature with var_smoothing=0 would leave a zero
            # variance; the floor keeps posteriors finite.
            X = [5.0 5.0; 5.0 5.0; 5.0 5.0; 5.0 5.0]
            y = [0, 0, 1, 1]
            gnb = GaussianNB(var_smoothing=0.0)
            gnb(X, y)
            probs = gnb(X; type=:probs)
            @test all(isfinite, probs)
            @test all(isapprox.(sum(probs, dims=2), 1.0; atol=1e-8))
            @test all(isfinite, gnb.var_)
            # Predictions are well-defined (no NaN propagating into argmax)
            @test all(p -> p in gnb.classes_, gnb(X))
        end

        @testset "show method" begin
            gnb = GaussianNB()
            buf = IOBuffer()
            show(buf, gnb)
            s = String(take!(buf))
            @test occursin("GaussianNB", s)
            @test occursin("fitted=false", s)
        end
    end

    @testset "MultinomialNB" begin
        using NovaML.NaiveBayes: MultinomialNB
        using NovaML: CountVectorizer

        @testset "Constructor defaults" begin
            mnb = MultinomialNB()
            @test mnb.alpha == 1.0
            @test mnb.fit_prior == true
            @test mnb.class_prior === nothing
            @test mnb.fitted == false
            @test isempty(mnb.classes_)
        end

        @testset "Constructor custom params" begin
            mnb = MultinomialNB(alpha=0.5, fit_prior=false, class_prior=[0.2, 0.8])
            @test mnb.alpha == 0.5
            @test mnb.fit_prior == false
            @test mnb.class_prior == [0.2, 0.8]
        end

        @testset "Constructor validation" begin
            @test_throws ArgumentError MultinomialNB(alpha=-0.1)
            # NaN/Inf pass a bare `< 0` check (both compare false) but would
            # propagate into NaN `feature_log_prob_`/posteriors at fit.
            @test_throws ArgumentError MultinomialNB(alpha=NaN)
            @test_throws ArgumentError MultinomialNB(alpha=Inf)
            # Value-only `class_prior` validation is data-independent, so it happens
            # in the constructor (fail fast), not just at fit. Length validation
            # needs the class count and stays in fit (see "Bad class_prior ...").
            # class_prior must be *strictly positive* (fit takes log.(class_prior)).
            @test_throws ArgumentError MultinomialNB(class_prior=[0.3, 0.3])    # sums to 0.6
            @test_throws ArgumentError MultinomialNB(class_prior=[0.0, 1.0])    # zero entry (log → -Inf)
            @test_throws ArgumentError MultinomialNB(class_prior=[-0.1, 1.1])   # negative entry
            @test_throws ArgumentError MultinomialNB(class_prior=[NaN, 0.5])    # non-finite entry
            @test_throws ArgumentError MultinomialNB(class_prior=[Inf, 0.5])    # non-finite entry
        end

        @testset "Predict not fitted" begin
            mnb = MultinomialNB()
            @test_throws ErrorException mnb([1.0, 0.0])
            @test_throws ErrorException mnb([1.0 0.0; 0.0 1.0])
        end

        @testset "Fit + hand-computed numeric correctness" begin
            # Two classes, two features, alpha = 1.0, fit_prior = true.
            # class 1 rows: [2,1] and [1,0]  -> feature_count = [3,1], total raw = 4
            # class 2 rows: [0,2] and [0,1]  -> feature_count = [0,3], total raw = 3
            X = [2.0 1.0; 1.0 0.0; 0.0 2.0; 0.0 1.0]
            y = [1, 1, 2, 2]
            mnb = MultinomialNB(alpha=1.0)
            mnb(X, y)

            @test mnb.fitted == true
            @test mnb.classes_ == [1, 2]
            @test mnb.feature_count_ == [3.0 1.0; 0.0 3.0]
            @test mnb.class_count_ == [2.0, 2.0]

            # feature_log_prob with Laplace smoothing (denominator = raw_total + alpha*n_features)
            #   class 1: log(4/6), log(2/6)   ; class 2: log(1/5), log(4/5)
            @test mnb.feature_log_prob_[1, 1] ≈ log(4 / 6)
            @test mnb.feature_log_prob_[1, 2] ≈ log(2 / 6)
            @test mnb.feature_log_prob_[2, 1] ≈ log(1 / 5)
            @test mnb.feature_log_prob_[2, 2] ≈ log(4 / 5)

            # fitted prior: both classes have 2 samples -> log(0.5)
            @test mnb.class_log_prior_ ≈ [log(0.5), log(0.5)]

            # Sample [1,0]:  jll1 = log(.5)+log(4/6) ;  jll2 = log(.5)+log(1/5)  -> class 1
            @test mnb([1.0, 0.0]) == 1
            # Sample [0,1]:  jll1 = log(.5)+log(2/6) ;  jll2 = log(.5)+log(4/5)  -> class 2
            @test mnb([0.0, 1.0]) == 2
        end

        @testset "Predict probabilities" begin
            X = [2.0 1.0; 1.0 0.0; 0.0 2.0; 0.0 1.0]
            y = [1, 1, 2, 2]
            mnb = MultinomialNB()
            mnb(X, y)

            probs = mnb(X; type=:probs)
            @test size(probs) == (4, 2)
            @test all(probs .>= 0)
            @test all(isapprox.(sum(probs, dims=2), 1.0; atol=1e-8))
        end

        @testset "fit_prior=false gives uniform prior" begin
            X = [2.0 1.0; 1.0 0.0; 1.0 0.0; 0.0 3.0]   # imbalanced classes
            y = [1, 1, 1, 2]
            mnb = MultinomialNB(fit_prior=false)
            mnb(X, y)
            @test mnb.class_log_prior_ ≈ [log(0.5), log(0.5)]
        end

        @testset "Bad class_prior length rejected at fit" begin
            X = [1.0 0.0; 0.0 1.0]
            y = [1, 2]
            # Wrong length is the genuine fit-time rejection: [0.5, 0.2, 0.3] is a
            # valid prior vector by value (sums to 1, all positive, finite) so it
            # clears the constructor, then fails the length check in fit against the
            # 2 classes. (Value-only failures are caught earlier, in the
            # constructor; see "Constructor validation".)
            @test_throws ArgumentError MultinomialNB(class_prior=[0.5, 0.2, 0.3])(X, y)
        end

        @testset "Mismatched X/y rows rejected" begin
            X = [1.0 0.0; 0.0 1.0; 1.0 1.0]
            @test_throws DimensionMismatch MultinomialNB()(X, [1, 2])
        end

        @testset "Empty / zero-feature data rejected" begin
            # No samples (0-row X with matching empty y): would otherwise produce a
            # degenerate zero-class "fitted" model.
            @test_throws ArgumentError MultinomialNB()(Matrix{Float64}(undef, 0, 2), Int[])
            # No features (0-column X): empty per-class log-probabilities.
            @test_throws ArgumentError MultinomialNB()(Matrix{Float64}(undef, 3, 0), [1, 2, 2])
        end

        @testset "Negative features rejected" begin
            X = [1.0 -1.0; 0.0 2.0]
            y = [1, 2]
            @test_throws ArgumentError MultinomialNB()(X, y)
        end

        @testset "Non-finite features rejected" begin
            # `NaN < 0` is false and `Inf >= 0`, so non-finite counts slip past the
            # non-negativity check; a NaN poisons feature_count_, and an Inf makes
            # log(Inf) - log(Inf) == NaN in feature_log_prob_. Reject at fit and at
            # predict (matching scikit-learn's check_array).
            y = [1, 1, 2, 2]
            @test_throws ArgumentError MultinomialNB()([2.0 1.0; NaN 0.0; 0.0 2.0; 0.0 1.0], y)
            @test_throws ArgumentError MultinomialNB()([2.0 1.0; Inf 0.0; 0.0 2.0; 0.0 1.0], y)

            # Fit on clean data, then reject non-finite predict inputs.
            X = [2.0 1.0; 1.0 0.0; 0.0 2.0; 0.0 1.0]
            mnb = MultinomialNB()
            mnb(X, y)
            @test_throws ArgumentError mnb([NaN, 0.0])
            @test_throws ArgumentError mnb([Inf, 0.0])
            @test_throws ArgumentError mnb([1.0 NaN; 0.0 1.0])
            @test_throws ArgumentError mnb([1.0 0.0; Inf 1.0])
        end

        @testset "alpha=0 produces no NaN" begin
            # A feature unseen in a class would give log P = -Inf, and 0 * -Inf = NaN
            # at predict time; clipping alpha to 1e-10 keeps everything finite.
            X = [2.0 0.0; 1.0 0.0; 0.0 2.0; 0.0 1.0]   # feature 2 unseen in class 1,
            y = [1, 1, 2, 2]                            # feature 1 unseen in class 2
            mnb = MultinomialNB(alpha=0.0)
            mnb(X, y)
            @test all(isfinite, mnb.feature_log_prob_)
            probs = mnb(X; type=:probs)
            @test all(isfinite, probs)
            @test all(isapprox.(sum(probs, dims=2), 1.0; atol=1e-8))
            # Single-sample predict is also NaN-free and recovers the obvious labels
            @test mnb([1.0, 0.0]) == 1
            @test mnb([0.0, 1.0]) == 2
        end

        @testset "Text classification with CountVectorizer" begin
            docs = [
                "football game goal striker",
                "goal match football striker",
                "election vote senate government",
                "senate vote election government"
            ]
            y = ["sport", "sport", "politics", "politics"]

            cv = CountVectorizer()
            X = cv(docs)                      # sparse count matrix

            mnb = MultinomialNB()
            mnb(X, y)

            @test mnb.fitted == true
            @test sort(mnb.classes_) == ["politics", "sport"]
            @test mnb.n_features_in_ == size(X, 2)

            # In-sample predictions recover the labels (well-separated vocabulary)
            preds = mnb(X)
            @test preds == y

            # Probabilities are valid
            probs = mnb(X; type=:probs)
            @test size(probs) == (4, 2)
            @test all(isapprox.(sum(probs, dims=2), 1.0; atol=1e-8))
        end

        @testset "show method" begin
            mnb = MultinomialNB()
            buf = IOBuffer()
            show(buf, mnb)
            s = String(take!(buf))
            @test occursin("MultinomialNB", s)
            @test occursin("fitted=false", s)
        end
    end
```

> Note: the `using ...` lines inside each new `@testset` mirror the existing
> pattern used by the `KNeighborsRegressor` and `GradientBoostingRegressor`
> testsets (which place their `using NovaML.X: Y` import at the top of the
> testset block). `CountVectorizer` is re-exported from the top-level `NovaML`
> module (`src/NovaML.jl:11`), so `using NovaML: CountVectorizer` is correct.

### Verification for Section 5

Run the full suite:

```bash
cd /Users/ilker/Documents/NovaML.jl && julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all testsets pass, including the new `GaussianNB` and `MultinomialNB`
groups, and the pre-existing `KNeighborsRegressor` / `GradientBoostingRegressor`
groups remain green (no regressions). The final summary should report `0` failures
and `0` errors.

---

## Section 6: Document the new models in `README.md`

### Modify `README.md`

The current `### Neighbors` section (lines 145–148) reads:

```markdown
### Neighbors

- `KNeighborsClassifier`: K-nearest neighbors classifier

### Decomposition
```

**Required change (this slice).** Insert a new `### NaiveBayes` subsection between
the `Neighbors` block and the `Decomposition` block, leaving the existing
`Neighbors` block untouched, so it becomes:

```markdown
### Neighbors

- `KNeighborsClassifier`: K-nearest neighbors classifier

### NaiveBayes

- `GaussianNB`: Gaussian Naive Bayes classifier for continuous features. Estimates per-class feature means and variances together with class priors, and classifies using Gaussian likelihoods computed in log space for numerical stability.
- `MultinomialNB`: Multinomial Naive Bayes classifier for count/frequency features (e.g. text data produced by `CountVectorizer` / `TfidfVectorizer`). Uses Laplace/Lidstone (`alpha`) smoothing and supports fitted, user-supplied, or uniform class priors.

### Decomposition
```

> **Optional, separate cleanup (not part of the NaiveBayes slice).** The
> `KNeighborsRegressor` already exists in the codebase
> (`src/Neighbors/KNeighborsRegressor.jl`) but is not listed under `### Neighbors`.
> Adding the bullet `- `KNeighborsRegressor`: K-nearest neighbors regressor` is an
> unrelated documentation fix; do it only as a deliberate, clearly-scoped extra,
> and it must not be required for this slice to be considered complete.

Optionally (recommended, not required), add a short usage example near the existing
LinearModel example. Place the following fenced block after the `### NaiveBayes`
bullet list:

````markdown
```julia
using NovaML.NaiveBayes: GaussianNB

gnb = GaussianNB()
gnb(Xtrn, ytrn)            # fit
ŷ = gnb(Xtst)             # predict labels
probs = gnb(Xtst; type=:probs)   # class posteriors
```
````

### Verification for Section 6

Automated, mechanical checks only (no manual rendered-Markdown inspection — any
human "does it read well" review belongs in the Operator Validation Checklist).

Confirm the new heading landed exactly once:

```bash
cd /Users/ilker/Documents/NovaML.jl && grep -c "^### NaiveBayes$" README.md
```

Expected: `1`.

Confirm both model bullets are present under the new section:

```bash
cd /Users/ilker/Documents/NovaML.jl && grep -c -E "^- \`(GaussianNB|MultinomialNB)\`:" README.md
```

Expected: `2`.

---

## Testing Strategy

All tests are automated (Section Authoring categories 1 and 2 only). They live in
`test/runtests.jl` and run via `julia --project=. -e 'using Pkg; Pkg.test()'`.

| Test group / case | Model | What it verifies | Category |
|---|---|---|---|
| Constructor defaults | both | Default hyperparameters and zero-initialized learned fields | 1 |
| Constructor custom params | both | Non-default keyword args stored on the struct | 1 |
| Constructor validation | both | Negative *and* non-finite (`NaN`/`Inf`) `var_smoothing` / `alpha`, **and** value-invalid `priors` / `class_prior` (sum≠1, negative/zero, non-finite), raise `ArgumentError` at construction | 1 |
| Predict not fitted | both | Vector and matrix predict throw `ErrorException` before fit | 1 |
| Fit sets learned attributes | GaussianNB | `classes_`, `theta_`, `var_`, priors, counts, `fitted=true`, returns self | 1 |
| Predict — well-separated classes | GaussianNB | In-sample labels recovered; single-sample predict | 1 |
| Predict probabilities | both | `type=:probs` returns non-negative matrix whose rows sum to 1 | 1 |
| Custom priors used | GaussianNB | Supplied `priors` propagate to `class_prior_` | 1 |
| Bad priors / class_prior length rejected at fit | both | Wrong-length (value-valid) priors raise `ArgumentError` at fit, where the class count is known | 1 |
| Mismatched X/y rows rejected | both | `length(y) != size(X,1)` raises `DimensionMismatch` | 1 |
| Mismatched feature count at predict rejected | GaussianNB | Predict input whose feature count differs from `n_features_in_` raises `DimensionMismatch` (closes the silent length-1 broadcast) | 1 |
| Empty / zero-feature data rejected | both | 0-row and 0-feature `X` raise `ArgumentError` at fit | 1 |
| Non-finite features rejected | both | `NaN`/`Inf` feature values raise `ArgumentError` at fit **and** predict (closes the silent NaN-poisoned mispredict that the negative/feature-count guards miss) | 1 |
| Negative features rejected | MultinomialNB | Negative feature values raise `ArgumentError` | 1 |
| `alpha=0` produces no NaN | MultinomialNB | Clipped `alpha` keeps `feature_log_prob_`/posteriors finite and predictions correct | 1 |
| Degenerate constant data — no NaN | GaussianNB | Variance floor keeps posteriors finite for constant data / `var_smoothing=0` | 1 |
| Fit + hand-computed numeric correctness | MultinomialNB | `feature_log_prob_`, `class_log_prior_`, and predictions match values computed by hand from a tiny count matrix | 1 |
| fit_prior=false uniform prior | MultinomialNB | Uniform `class_log_prior_` when `fit_prior=false` | 1 |
| Text classification with CountVectorizer | MultinomialNB | Fits on a sparse `CountVectorizer` matrix and recovers labels in-sample | 1 |
| show method | both | `Base.show` includes the type name and `fitted=` | 1 |
| Full suite green | both | `Pkg.test()` reports 0 failures / 0 errors incl. pre-existing groups | 2 |

All numeric expectations are derived from small, deterministic inputs (no RNG):
the `GaussianNB` separability test uses two tightly clustered, far-apart groups; the
`MultinomialNB` correctness test uses a 4×2 integer-valued count matrix whose
smoothed log-probabilities and posterior argmax are computed by hand in the test
comments.

---

## Operator Validation Checklist

This section is owned by the human operator, not by the creator
subprocess. Maestro's implementation reviewer MUST NOT escalate a
section solely because items in this checklist read
`pending operator validation`. Maestro's final reviewer MUST allow
`Implementation Complete: YES` when all creator-performable work
and all automated checks have passed, even if items here remain
`pending operator validation`.

Each row has one of three states:

- `pending operator validation` — the action has not been performed
  yet. This is the default state. It is HONEST and ALLOWED. Do not
  rewrite it to `passed` unless the operator has actually performed
  and recorded the action.
- `passed` — the operator performed the action and recorded the
  evidence (date, host, screenshot path, log path, or commit SHA).
  The evidence MUST be specific. A bare `passed` with no captured
  artifact is treated as fabricated.
- `waived by operator` — the operator decided the action is not
  required for this slice and recorded the reason inline.

### Manual smoke

- [ ] `pending operator validation` — `julia --project=.` REPL walkthrough:
  `using NovaML.NaiveBayes`, fit `GaussianNB` on a small continuous dataset,
  call predict and `type=:probs`, and sanity-check that the labels and
  probabilities look reasonable. Evidence: `<terminal log or note when run>`.
- [ ] `pending operator validation` — `julia --project=.` REPL walkthrough:
  build a `CountVectorizer` matrix from a few text documents, fit `MultinomialNB`,
  predict, and confirm the classification is sensible. Evidence:
  `<terminal log or note when run>`.
- [ ] `pending operator validation` — Confirm no regression in existing
  submodules by loading `using NovaML` and instantiating a previously-existing
  model (e.g. `KNeighborsClassifier`, `GradientBoostingRegressor`). Evidence:
  `<terminal log or note when run>`.

### External / live-host checks

- (none required for this slice)

### Commit / release hygiene

- [ ] `pending operator validation` — Commit the final shipped state of the files
  in this slice (`src/NaiveBayes/NaiveBayes.jl`, `src/NaiveBayes/GaussianNB.jl`,
  `src/NaiveBayes/MultinomialNB.jl`, `src/NovaML.jl`, `test/runtests.jl`,
  `README.md`, `docs/implementations/IMPLEMENTATION_NAIVE_BAYES.md`).
  Evidence: `<commit SHA when landed>`.
- [ ] `pending operator validation` — Bump the package `version` in `Project.toml`
  (currently `0.5.0`) if a release is intended, and tag it. Evidence:
  `<tag name / new version when created>`.
