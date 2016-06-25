#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)
require(infotheo, quietly=TRUE)

win = 500
input_neurons = 256
target_rate = 10

c(spikes, epoch) := read.spikes.wd()

sp = spikes
sp$values = sp$values[-(1:input_neurons)]

input_spikes = spikes


rv = get.rate.vectors(sp, win)

max_t = spikes.list.max.t(sp)
mean_rate = mean(sapply(sp$values, length)) / (max_t/1000)

rect = function(x) if (x>0) {x} else {0.0}
rate_denom = 1+rect(exp(mean_rate-target_rate)*1e-02-1e-02)

df = as.data.frame(sapply(1:nrow(rv$values), function(i) as.integer(win*rv$values[i, ]) ))

cat(- entropy(df)/rate_denom, "\n")
