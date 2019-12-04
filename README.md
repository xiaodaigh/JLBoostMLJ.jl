# MLJJLBoost.jl

The [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) interface to [JLBoost.jl](https://github.com/xiaodaigh/JLBoost.jl) a hackable implementation of Gradient Boosting Regression Trees.


## Usage Example

````julia
using RDatasets
iris = dataset("datasets", "iris")
iris[!, :is_setosa] .= iris.Species .== "setosa"

using MLJ, MLJBase, MLJJLBoost
X, y = unpack(iris, x->!(x in [:is_setosa, :Species]), ==(:is_setosa))

using MLJJLBoost:JLBoostClassifier
model = JLBoostClassifier()
````


````
JLBoostClassifier(loss = JLBoost.LogitLogLoss(),
                  nrounds = 1,
                  subsample = 1.0,
                  eta = 1.0,
                  max_depth = 6,
                  min_child_weight = 1.0,
                  lambda = 0.0,
                  gamma = 0.0,
                  colsample_bytree = 1,) @ 1…09
````





### Simple Fitting

Fit the model
````julia
mljmodel = fit(model, 1, X, y)
````


````
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
(feature = :PetalLength, split_at = 1.9, cutpt = 50, gain = 133.33333333333
334, lweight = 2.0, rweight = -2.0)
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
(fitresult = (treemodel = JLBoost.JLBoostTrees.JLBoostTreeModel(JLBoost.JLB
oostTrees.JLBoostTree[
   -- PetalLength <= 1.9
     ---- weight = 2.0

   -- PetalLength > 1.9
     ---- weight = -2.0
], JLBoost.LogitLogLoss(), :__y__),
              target_levels = Bool[0, 1],),
 cache = nothing,
 report = (AUC = 0.16666666666666669,
           feature_importance = 1×4 DataFrame
│ Row │ feature     │ Quality_Gain │ Coverage │ Frequency │
│     │ Symbol      │ Float64      │ Float64  │ Float64   │
├─────┼─────────────┼──────────────┼──────────┼───────────┤
│ 1   │ PetalLength │ 1.0          │ 1.0      │ 1.0       │,),)
````




Predicting using the model

````julia
predict(model, mljmodel.fitresult, X)
````


````
150-element Array{UnivariateFinite{Bool,UInt32,Float64},1}:
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 ⋮                                          
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
````





Feature Importance for simple fitting
One can obtain the feature importance using the `feature_importance` function

````julia
feature_importance(mljmodel.fitresult.treemodel, X, y)
````


````
1×4 DataFrame
│ Row │ feature     │ Quality_Gain │ Coverage │ Frequency │
│     │ Symbol      │ Float64      │ Float64  │ Float64   │
├─────┼─────────────┼──────────────┼──────────┼───────────┤
│ 1   │ PetalLength │ 1.0          │ 1.0      │ 1.0       │
````





### Using MLJ machines

Put the model and data in a machine

````julia
mljmachine  = machine(model, X, y)
````


````
Machine{JLBoostClassifier} @ 8…76
````





Fit model using machine

````julia
fit!(mljmachine)
````


````
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
(feature = :PetalLength, split_at = 1.9, cutpt = 50, gain = 133.33333333333
334, lweight = 2.0, rweight = -2.0)
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
Machine{JLBoostClassifier} @ 8…76
````





Predict using machine

````julia
predict(mljmachine, X)
````


````
150-element Array{UnivariateFinite{Bool,UInt32,Float64},1}:
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 UnivariateFinite(false=>0.881, true=>0.119)
 ⋮                                          
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
 UnivariateFinite(false=>0.119, true=>0.881)
````





Feature importance using machine

````julia
feature_importance(fitted_params(mljmachine).fitresult, X, y)
````


````
1×4 DataFrame
│ Row │ feature     │ Quality_Gain │ Coverage │ Frequency │
│     │ Symbol      │ Float64      │ Float64  │ Float64   │
├─────┼─────────────┼──────────────┼──────────┼───────────┤
│ 1   │ PetalLength │ 1.0          │ 1.0      │ 1.0       │
````


