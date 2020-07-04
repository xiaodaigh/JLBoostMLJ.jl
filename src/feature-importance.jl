import JLBoost: feature_importance

import MLJBase: CrossEntropy

CrossEntropy(x, y::CategoricalVector) = CrossEntropy(x, y.refs .- 1)
