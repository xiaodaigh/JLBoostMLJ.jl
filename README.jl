
using RDatasets;
iris = dataset("datasets", "iris");
iris[!, :is_setosa] .= iris.Species .== "setosa";

using MLJ, JLBoostmlj;
X, y = unpack(iris, x->!(x in [:is_setosa, :Species]), ==(:is_setosa));

using JLBoostmlj:JLBoostClassifier;
model = JLBoostClassifier()


mljmachine  = machine(model, X, y)


fit!(mljmachine)


predict(mljmachine, X)


feature_importance(fitted_params(mljmachine).fitresult, X, y)


using CategoricalArrays
y_cate = categorical(y)


using JLBoost, JLBoostmlj, MLJ
jlb = JLBoostClassifier()
r1 = range(jlb, :nrounds, lower=1, upper = 6)
r2 = range(jlb, :max_depth, lower=1, upper = 6)
r3 = range(jlb, :eta, lower=0.1, upper=1.0)


tm = TunedModel(model = jlb, ranges = [r1, r2, r3], measure = cross_entropy)
m = machine(tm, X, y_cate)


fit!(m)


fitted_params(m).best_model.max_depth
fitted_params(m).best_model.nrounds
fitted_params(m).best_model.eta


mljmodel = fit(model, 1, X, y)


predict(model, mljmodel.fitresult, X)


feature_importance(mljmodel.fitresult.treemodel, X, y)

