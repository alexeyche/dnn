#!/usr/bin/env Rscript

require(Rdnn, quietly=TRUE)
require(infotheo, quietly=TRUE)

win = 100
input_neurons = 256

c(spikes, epoch) := read.spikes.wd()
sp = spikes
sp$values = sp$values[-(1:input_neurons)]


rv = get.rate.vectors(sp, win)
df = as.data.frame(sapply(1:nrow(rv$values), function(i) as.integer(win*rv$values[i, ]) ))
cat(-entropy(df), "\n")
