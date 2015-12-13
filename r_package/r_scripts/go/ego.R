#!/usr/bin/env Rscript

require(methods)
require(Rdnn)
require(DiceOptim)


opts = list()
opts[["--run-name"]] = list(
    description="Name of run from runs dir"
)
opts[["--const"]] = list(
    description="Const json for simulations"
)
opts[["--var-specs"]] = list(
    description="Variable specifications"
)
opts[["--design-points"]] = list(
    description = "Number of design points to starts with",
    default = 10,
    process = as.numeric
)
opts[["--dim-size"]] = list(
    description = "Size of dimension to deal",
    default = 7,
    process = as.numeric
)
opts[["--spike-input"]] = list(
    description = "Spike input"
)
opts[["--evaluation-data"]] = list(
    description = "Evaluation data"
)


args <- commandArgs()
c(run_name, const, var_specs, design_points, dim_size, spike_input, evaluation_data) := 
    parse.options(args, opts)

id = 1

eval = function(x, sim_jobs=7) {
    cmd_line = sprintf(paste(
        run.evolve.script()
      , "-c %s"
      , "-v %s"
      , "-a id=%s"
      , "--tag %s"
      , "SimpleRunner"
      , "--spike-input %s"
      , "--evaluation-data %s"
      , "--sim-jobs %s"
    ), const, var_specs, id, run_name, spike_input, evaluation_data, sim_jobs)
    r = system(cmd_line, intern = TRUE, ignore.stderr=TRUE, input=paste(x, collapse=" "))
    id <<- id + 1
    return(as.numeric(r))
}

X = optimumLHS(design_points, dim_size)

Y = NULL
for(ri in 1:nrow(X)) {
    Y = c(Y, eval(X[ri, ]))
}

m = km(
    ~ .
    , design = as.data.frame(X)
    , response = data.frame(Y=Y)
    , optim.method = "gen"
    , control = list(pop.size = 100, max.generations = 30, wait.generations = 5, BFGSburnin = 2)
)

EGO.nsteps(m, eval, 1000, lower=rep(0, ncol(X)), upper=rep(1, ncol(X)))

