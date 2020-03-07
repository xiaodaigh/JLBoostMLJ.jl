# Weave readme
using Pkg
Pkg.activate("c:/git/JLBoostMLJ")

Pkg.add("Weave")
using Weave

weave("c:/git/JLBoostMLJ/README.jmd", out_path="c:/git/JLBoostMLJ", doctype="github")

Pkg.rm("Weave")


if false
tangle("c:/git/JLBoostMLJ/README.jmd")
end
