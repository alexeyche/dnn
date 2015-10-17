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
    default = 120,
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
opts[["--prolongation"]] = list(
    description="Time to which spikes list must be created",
    default = 5000,
    process = as.integer
)
args <- commandArgs()

c(neurons, dst_file, dt, sample_size, ts_name, sample_gap, prolongation) :=
    parse.options(args, opts)




#sel=c(1:10, 50:60, 100:110, 150:160, 200:210, 250:260)
#sapply(1:6, function(x) sample((x-1)*50+1:50, 1))
sel=c(38, 89, 137, 163, 244, 285)

c(train_ts, test_ts) := prepare.ucr.data(sample_size, ts_name, gap_between_patterns = 0, sel=sel)


res_sp = empty.spikes(neurons)
while(tail(sp$ts_info$labels_timeline, n=1) < prolongation) {
    sp = intercept.data.to.spikes(
        train_ts
        , neurons
        , 1
        , dt
        , sample_gap
    )
    res_sp = cat.spikes(res_sp, sp)
}


proto.write(sp, dst_file)


