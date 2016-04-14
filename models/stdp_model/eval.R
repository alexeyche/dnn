#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)

input_neurons = 100

last_epoch = as.numeric(strsplit(system("ls -t *.pb | head -n 1", intern=TRUE), "_")[[1]][1])

eval_spikes_fname = sprintf("%d_eval_spikes.pb", last_epoch)
spikes_fname = sprintf("%d_spikes.pb", last_epoch)

spikes = NULL
if (file.exists(eval_spikes_fname)) {
    spikes = proto.read(eval_spikes_fname)
} else 
if (file.exists(spikes_fname)) {
    spikes = proto.read(spikes_fname)
} else {
    stop(sprintf("Failed to find spikes in directory %d", getwd()))
}

rates = NULL
labs = NULL
for (sp in chop.spikes.list(spikes)) {
    rates = rbind(rates, sapply(sp$values[-(1:input_neurons)], length)/sp$info[[1]]$duration)
    labs = c(labs, sp$info[[1]]$label)
}
K = rates %*% t(rates)
colnames(K) <- labs
c(y, M, N, A) := KFD(K)
metric = -tr(M)/tr(N)

ans = K %*% y[, 1:2]

png(sprintf("%d_eval.png", last_epoch), width=1024, height=768)
par(mfrow=c(1,2))

metrics_str = sprintf("%f", metric)
plot(Re(ans[,1]), col=as.integer(colnames(K)), main=metrics_str) 
plot(Re(ans), col=as.integer(colnames(K)))        

cat(metric, "\n") 

