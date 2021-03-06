# I have decided NOT to maintain this package any further. Please do NOT use it.

# JLBoostMLJ.jl

The [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) interface to [JLBoost.jl](https://github.com/xiaodaigh/JLBoost.jl), a hackable implementation of Gradient Boosting Regression Trees.


## Usage Example

````julia

using RDatasets;
iris = dataset("datasets", "iris");
iris[!, :is_setosa] = iris.Species .== "setosa";

using MLJ, JLBoostMLJ;
X, y = unpack(iris, x->!(x in [:is_setosa, :Species]), ==(:is_setosa));

using JLBoostMLJ:JLBoostClassifier;
model = JLBoostClassifier()
````


````
JLBoostClassifier(
    loss = JLBoost.LogitLogLoss(),
    nrounds = 1,
    subsample = 1.0,
    eta = 1.0,
    max_depth = 6,
    min_child_weight = 1.0,
    lambda = 0.0,
    gamma = 0.0,
    colsample_bytree = 1) @087
````





### Using MLJ machines

Put the model and data in a machine

````julia

mljmachine  = machine(model, X, y)
````


````
Machine{JLBoostClassifier} @730 trained 0 times.
  args: 
    1:	Source @910 ⏎ `ScientificTypes.Table{AbstractArray{ScientificTypes.C
ontinuous,1}}`
    2:	Source @954 ⏎ `AbstractArray{ScientificTypes.Count,1}`
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
Machine{JLBoostClassifier} @730 trained 1 time.
  args: 
    1:	Source @910 ⏎ `ScientificTypes.Table{AbstractArray{ScientificTypes.C
ontinuous,1}}`
    2:	Source @954 ⏎ `AbstractArray{ScientificTypes.Count,1}`
````





Predict using machine

````julia

predict(mljmachine, X)
````


````
150-element Array{MLJBase.UnivariateFinite{ScientificTypes.Multiclass{2},Bo
ol,UInt32,Float64},1}:
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 ⋮
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
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





#### Hyperparameter tuning

Data preparation: need to convert `y` to categorical

````julia

y_cate = categorical(y)
````


````
150-element CategoricalArrays.CategoricalArray{Bool,1,UInt32}:
 true
 true
 true
 true
 true
 true
 true
 true
 true
 true
 ⋮
 false
 false
 false
 false
 false
 false
 false
 false
 false
````





Set up some hyperparameter ranges

````julia

using JLBoost, JLBoostMLJ, MLJ
jlb = JLBoostClassifier()
r1 = range(jlb, :nrounds, lower=1, upper = 6)
r2 = range(jlb, :max_depth, lower=1, upper = 6)
r3 = range(jlb, :eta, lower=0.1, upper=1.0)
````


````
MLJBase.NumericRange(Float64, :eta, ... )
````





Set up the machine
````julia

tm = TunedModel(model = jlb, ranges = [r1, r2, r3], measure = cross_entropy)
m = machine(tm, X, y_cate)
````


````
Machine{ProbabilisticTunedModel{Grid,…}} @109 trained 0 times.
  args: 
    1:	Source @664 ⏎ `ScientificTypes.Table{AbstractArray{ScientificTypes.C
ontinuous,1}}`
    2:	Source @788 ⏎ `AbstractArray{ScientificTypes.Multiclass{2},1}`
````





Fit it!
````julia

fit!(m)
````


````
Machine{ProbabilisticTunedModel{Grid,…}} @109 trained 1 time.
  args: 
    1:	Source @664 ⏎ `ScientificTypes.Table{AbstractArray{ScientificTypes.C
ontinuous,1}}`
    2:	Source @788 ⏎ `AbstractArray{ScientificTypes.Multiclass{2},1}`
````





Inspected the tuned parameters
````julia

fitted_params(m).best_model.max_depth
fitted_params(m).best_model.nrounds
fitted_params(m).best_model.eta
````


````
0.9
````





### Simple Fitting

Fit the model with `verbosity = 1`
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
oostTrees.AbstractJLBoostTree[eta = 1.0 (tree weight)

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
150-element Array{MLJBase.UnivariateFinite{ScientificTypes.Multiclass{2},Bo
ol,UInt32,Float64},1}:
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.881, true=>0.119)
 ⋮
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
 UnivariateFinite{ScientificTypes.Multiclass{2}}(false=>0.119, true=>0.881)
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


