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
    default = spikes.path("intercept_spikes.pb")
)
opts[["--dt"]] = list(
    description="delta t for spikes",
    default = 5,
    process = as.numeric
)
opts[["--sample-size"]] = list(
    description="size of one sample",
    default = 500,
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
    description="Till what time do we need to prolongate time series if it's not long enough",
    default = 60000*4,
    process = as.integer
)

args <- commandArgs()

c(neurons, dst_file, dt, sample_size, ts_name, sample_gap, prolongation) :=
    parse.options(args, opts)


#sel=c(1:10, 50:60, 100:110, 150:160, 200:210, 250:260)
#sapply(1:6, function(x) sample((x-1)*50+1:50, 1))
#sel=c(38, 89, 137, 163, 244, 285)

test_ts = time.series(
    matrix(c(1:100, 100:1), nrow=1)
  , ts.info(c("1","2"), c(100, 200))
)

#c(train_ts, test_ts) := prepare.ucr.data(sample_size, ts_name, gap_between_patterns = 0, sel=sel)

ts = test_ts

res_sp = empty.spikes(neurons)

while(TRUE) {
    sp = intercept.data.to.spikes(
        ts
        , neurons
        , 1
        , dt
        , sample_gap
    )
    res_sp = cat.spikes(res_sp, sp)
    if(tail(res_sp$ts_info$labels_timeline, n=1)>prolongation) {
        break
    }
}

proto.write(res_sp, dst_file)


