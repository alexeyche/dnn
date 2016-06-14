#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)
require(infotheo, quietly=TRUE)

win = 25
input_neurons = 256

c(spikes, epoch) := read.spikes.wd()

sp = spikes
sp$values = sp$values[-(1:input_neurons)]

input_spikes = spikes
input_spikes$values = input_spikes$values[1:input_neurons]

rv = get.rate.vectors(sp, win)
inp_rv = get.rate.vectors(input_spikes, win)

df = as.data.frame(sapply(1:nrow(rv$values), function(i) as.integer(win*rv$values[i, ]) ))
inp_df = as.data.frame(sapply(1:nrow(inp_rv$values), function(i) as.integer(win*inp_rv$values[i, ]) ))

mean_spikes = mean(sapply(spikes$values, length))

cat(-mutinformation(inp_df, df)/mean_spikes, "\n")
