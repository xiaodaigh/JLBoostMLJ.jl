# Weave readme
using Pkg
Pkg.activate("c:/git/MLJJLBoost")
using Weave

weave("c:/git/MLJJLBoost/README.jmd", out_path=:pwd, doctype="github")

if false


end
