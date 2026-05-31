using NovaML
using Test

@testset "NovaML.jl" begin

    @testset "KNeighborsRegressor" begin
        using NovaML.Neighbors: KNeighborsRegressor

        @testset "Constructor defaults" begin
            reg = KNeighborsRegressor()
            @test reg.n_neighbors == 5
            @test reg.weights == :uniform
            @test reg.algorithm == :auto
            @test reg.leaf_size == 30
            @test reg.p == 2
            @test reg.fitted == false
            @test reg.X === nothing
            @test reg.y === nothing
        end

        @testset "Constructor custom params" begin
            reg = KNeighborsRegressor(n_neighbors=3, weights=:distance, p=1, metric="manhattan")
            @test reg.n_neighbors == 3
            @test reg.weights == :distance
        end

        @testset "Constructor validation" begin
            @test_throws AssertionError KNeighborsRegressor(n_neighbors=0)
            @test_throws AssertionError KNeighborsRegressor(n_neighbors=-1)
            @test_throws AssertionError KNeighborsRegressor(weights=:invalid)
            @test_throws AssertionError KNeighborsRegressor(leaf_size=0)
            @test_throws AssertionError KNeighborsRegressor(p=0)
        end

        @testset "Fit" begin
            reg = KNeighborsRegressor(n_neighbors=2)
            X_train = [1.0 0.0; 2.0 0.0; 3.0 0.0; 4.0 0.0]
            y_train = [1.0, 2.0, 3.0, 4.0]
            reg(X_train, y_train)

            @test reg.fitted == true
            @test reg.n_features_in_ == 2
            @test reg.n_samples_fit_ == 4
            @test reg.X == X_train
            @test reg.y == y_train
        end

        @testset "Predict not fitted" begin
            reg = KNeighborsRegressor()
            @test_throws ErrorException reg(Float64[1.0 0.0])
        end

        @testset "Predict uniform weights" begin
            reg = KNeighborsRegressor(n_neighbors=2)
            X_train = [1.0 0.0; 2.0 0.0; 3.0 0.0; 4.0 0.0]
            y_train = [10.0, 20.0, 30.0, 40.0]
            reg(X_train, y_train)

            # Query point [1.5, 0.0] — two nearest are [1,0] (y=10) and [2,0] (y=20)
            preds = reg(Float64[1.5 0.0])
            @test length(preds) == 1
            @test preds[1] ≈ 15.0  # mean(10, 20)

            # Query point [3.5, 0.0] — two nearest are [3,0] (y=30) and [4,0] (y=40)
            preds = reg(Float64[3.5 0.0])
            @test preds[1] ≈ 35.0  # mean(30, 40)
        end

        @testset "Predict distance weights" begin
            reg = KNeighborsRegressor(n_neighbors=2, weights=:distance)
            X_train = [0.0 0.0; 10.0 0.0]
            y_train = [0.0, 10.0]
            reg(X_train, y_train)

            # Query point [1.0, 0.0]: dist to [0,0]=1, dist to [10,0]=9
            # w1 = 1/1 = 1, w2 = 1/9 ≈ 0.111
            # pred = (1*0 + 0.111*10) / (1 + 0.111) = 1.111/1.111 ≈ 1.0
            # (Using exact: w1=1/(1+eps), w2=1/(9+eps), result ≈ 1.0)
            preds = reg(Float64[1.0 0.0])
            @test preds[1] ≈ 1.0 atol=0.1

            # Query point [9.0, 0.0]: dist to [0,0]=9, dist to [10,0]=1
            # Closer to 10.0 → prediction should be near 9.0
            preds = reg(Float64[9.0 0.0])
            @test preds[1] ≈ 9.0 atol=0.1
        end

        @testset "Predict distance weights exact match" begin
            reg = KNeighborsRegressor(n_neighbors=2, weights=:distance)
            X_train = [0.0 0.0; 10.0 0.0]
            y_train = [5.0, 50.0]
            reg(X_train, y_train)

            # Query exactly on a training point: distance=0, w=1/eps() dominates
            preds = reg(Float64[0.0 0.0])
            @test preds[1] ≈ 5.0 atol=1e-10
        end

        @testset "Predict multiple samples" begin
            reg = KNeighborsRegressor(n_neighbors=1)
            X_train = [0.0 0.0; 1.0 0.0; 2.0 0.0]
            y_train = [100.0, 200.0, 300.0]
            reg(X_train, y_train)

            X_test = [0.1 0.0; 0.9 0.0; 2.1 0.0]
            preds = reg(X_test)
            @test length(preds) == 3
            @test preds[1] ≈ 100.0  # nearest to [0,0]
            @test preds[2] ≈ 200.0  # nearest to [1,0]
            @test preds[3] ≈ 300.0  # nearest to [2,0]
        end

        @testset "Fit returns self" begin
            reg = KNeighborsRegressor()
            X_train = [1.0 0.0; 2.0 0.0]
            y_train = [1.0, 2.0]
            result = reg(X_train, y_train)
            @test result === reg
        end

        @testset "Integer y converted to Float64" begin
            reg = KNeighborsRegressor(n_neighbors=1)
            X_train = [1.0 0.0; 2.0 0.0]
            y_train = [1, 2]  # integers
            reg(X_train, y_train)
            @test eltype(reg.y) == Float64
        end

        @testset "show method" begin
            reg = KNeighborsRegressor(n_neighbors=3)
            buf = IOBuffer()
            show(buf, reg)
            s = String(take!(buf))
            @test occursin("KNeighborsRegressor", s)
            @test occursin("n_neighbors=3", s)
            @test occursin("fitted=false", s)
        end
    end

    @testset "GradientBoostingRegressor" begin
        using NovaML.Ensemble: GradientBoostingRegressor
        using Random

        @testset "Constructor defaults" begin
            gbr = GradientBoostingRegressor()
            @test gbr.loss == "squared_error"
            @test gbr.learning_rate == 0.1
            @test gbr.n_estimators == 100
            @test gbr.subsample == 1.0
            @test gbr.max_depth == 3
            @test gbr.alpha == 0.9
            @test gbr.fitted == false
        end

        @testset "Constructor validation" begin
            @test_throws ArgumentError GradientBoostingRegressor(loss="invalid")
            @test_throws ArgumentError GradientBoostingRegressor(init="bad")
            @test_throws ArgumentError GradientBoostingRegressor(learning_rate=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(learning_rate=1.5)
            @test_throws ArgumentError GradientBoostingRegressor(n_estimators=0)
            @test_throws ArgumentError GradientBoostingRegressor(subsample=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(subsample=1.5)
            @test_throws ArgumentError GradientBoostingRegressor(validation_fraction=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(validation_fraction=1.0)
            @test_throws ArgumentError GradientBoostingRegressor(alpha=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(alpha=1.0)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_split=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_split=1.5)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_split=1)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_leaf=0.0)
            # NovaML caps float min_samples_leaf at 0.5 (values > 0.5 make splits impossible)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_leaf=0.6)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_leaf=0)
        end

        @testset "Fit and predict — squared_error" begin
            Random.seed!(42)
            X_train = randn(100, 3)
            y_train = 2.0 .* X_train[:, 1] .+ 0.5 .* X_train[:, 2] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            @test gbr.n_estimators_ == 50
            @test length(gbr.estimators_) == 50
            @test length(gbr.train_score_) == 50

            # Train loss decreases over boosting rounds
            @test gbr.train_score_[end] < gbr.train_score_[1]

            # Predictions are numeric vector
            preds = gbr(X_train)
            @test length(preds) == 100
            @test eltype(preds) <: AbstractFloat

            # Feature importances
            @test gbr.feature_importances_ !== nothing
            @test length(gbr.feature_importances_) == 3
            @test all(gbr.feature_importances_ .>= 0)
        end

        @testset "Fit and predict — absolute_error" begin
            Random.seed!(123)
            X_train = randn(80, 2)
            y_train = 3.0 .* X_train[:, 1] .+ randn(80) .* 0.2

            gbr = GradientBoostingRegressor(
                loss="absolute_error",
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                random_state=123
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            @test gbr.train_score_[end] < gbr.train_score_[1]

            preds = gbr(X_train)
            @test length(preds) == 80
            @test eltype(preds) <: AbstractFloat
        end

        @testset "Fit and predict — huber" begin
            Random.seed!(456)
            X_train = randn(80, 2)
            y_train = 1.5 .* X_train[:, 1] .- 0.5 .* X_train[:, 2] .+ randn(80) .* 0.3

            gbr = GradientBoostingRegressor(
                loss="huber",
                alpha=0.9,
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                random_state=456
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            @test gbr.train_score_[end] < gbr.train_score_[1]

            preds = gbr(X_train)
            @test length(preds) == 80
            @test eltype(preds) <: AbstractFloat
        end

        @testset "Early stopping" begin
            Random.seed!(789)
            X_train = randn(200, 2)
            y_train = X_train[:, 1] .+ randn(200) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.5,
                max_depth=3,
                n_iter_no_change=10,
                tol=1e-4,
                validation_fraction=0.2,
                random_state=789
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            # Should stop before reaching 500 estimators
            @test gbr.n_estimators_ < 500
        end

        @testset "Predict not fitted" begin
            gbr = GradientBoostingRegressor()
            @test_throws ErrorException gbr(randn(5, 2))
        end

        @testset "Subsample" begin
            Random.seed!(101)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ X_train[:, 2] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                subsample=0.8,
                learning_rate=0.1,
                max_depth=3,
                random_state=101
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            preds = gbr(X_train)
            @test length(preds) == 100
        end

        @testset "Warm start continuation" begin
            Random.seed!(202)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ X_train[:, 2] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                warm_start=true,
                random_state=202
            )
            gbr(X_train, y_train)
            @test gbr.fitted == true
            @test length(gbr.estimators_) == 30

            # Continue training with more estimators
            gbr.n_estimators = 60
            gbr(X_train, y_train)
            @test length(gbr.estimators_) == 60
            @test gbr.n_estimators_ == 60
            @test length(gbr.train_score_) == 60
        end

        @testset "Warm start — lowering n_estimators errors" begin
            Random.seed!(303)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                warm_start=true,
                random_state=303
            )
            gbr(X_train, y_train)
            @test length(gbr.estimators_) == 30

            # Lowering n_estimators with warm_start should throw
            gbr.n_estimators = 20
            @test_throws ArgumentError gbr(X_train, y_train)
        end

        @testset "Warm start — same n_estimators is no-op" begin
            Random.seed!(404)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                warm_start=true,
                random_state=404
            )
            gbr(X_train, y_train)
            preds1 = gbr(X_train)

            # Re-fit with same n_estimators — should keep same estimators
            gbr(X_train, y_train)
            preds2 = gbr(X_train)
            @test length(gbr.estimators_) == 30
            @test preds1 == preds2
        end
    end

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
        using NovaML.FeatureExtraction: CountVectorizer

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

end
