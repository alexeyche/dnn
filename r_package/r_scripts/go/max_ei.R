#!/usr/bin/env Rscript

require(methods)
require(Rdnn)
require(DiceOptim)

opts = list()
opts[["--run-name"]] = list(
    description="Name of run from runs dir"
)

args <- commandArgs()
run_name := parse.options(args, opts)

o = system(sprintf("%s %s", read.state.script(), runs.path(run_name)), intern=TRUE)
M = t(sapply(strsplit(o, ","), as.numeric))

X = M[, 1:(ncol(M)-1)]
Y = M[, ncol(M)]
Y = -log(-Y)
colnames(X) <- sprintf("x%d", 1:ncol(X))

m = km(
    ~ .
    , design = as.data.frame(X)
    , response = data.frame(Y=Y)
    , optim.method = "gen"
    , control = list(pop.size = 100, max.generations = 30, wait.generations = 5, BFGSburnin = 2)
)

p = max_EI(
    m
  , lower=rep(0, ncol(X)), upper=rep(1, ncol(X))
)

