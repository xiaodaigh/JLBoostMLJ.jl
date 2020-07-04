# Weave readme
using Pkg
cd("c:/git/JLBoostMLJ")
# Pkg.activate("c:/git/JLBoostMLJ")

Pkg.add("RDatasets")
Pkg.add("Weave")
using Weave

weave("c:/git/JLBoostMLJ/README.jmd", out_path="c:/git/JLBoostMLJ", doctype="github")

Pkg.rm("Weave")
Pkg.rm("RDatasets")


if false
tangle("c:/git/JLBoostMLJ/README.jmd")
end
