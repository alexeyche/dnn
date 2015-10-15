#!/usr/bin/env Rscript

library(methods)
require(Rdnn)

opts = list()
opts[["--neurons"]] = list(
    description="Number of neurons",
    default = 100,
    process = as.integer
)
opts[["--dst-file"]] = list(
    description="destination file for spikes",
    default = "intercept_spikes.pb"
)
opts[["--dt"]] = list(
    description="delta t for spikes",
    default = 1,
    process = as.numeric
)
opts[["--sample-size"]] = list(
    description="size of one sample",
    default = 1000,
    process = as.integer
)
opts[["--ts-name"]] = list(
    description="Name of time series from ucr",
    default = UCR.SYNTH
)
opts[["--sample-gap"]] = list(
    description="Gap between samples",
    default = 100,
    process = as.numeric
)

args <- commandArgs()

c(neurons, dst_file, dt, sample_size, ts_name, sample_gap) := parse.options(args, opts)




sel=c(1:10, 50:60, 100:110, 150:160, 200:210, 250:260)
c(train_ts, test_ts) := prepare.ucr.data(sample_size, ts_name, gap_between_patterns = 0, sel=sel)

sp = intercept.data.to.spikes(
    train_ts
  , neurons
  , 1
  , dt
  , sample_gap
)

proto.write(sp, dst_file)


