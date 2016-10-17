#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)

input_neurons = 256

epoch=as.numeric(strsplit(system(sprintf("basename $(ls -t %s/*.pb | head -n 1)", getwd()), intern=TRUE), "_")[[1]][1])
spikes = proto.read(sprintf("%d_spikes.pb", epoch))

rates = NULL
labs = NULL
for (sp in chop.spikes.list(spikes)) {
    rates = rbind(rates, sapply(sp$values[-(1:input_neurons)], length)/sp$info[[1]]$duration)
    labs = c(labs, sp$info[[1]]$label)
}
K = rates %*% t(rates)
colnames(K) <- labs
c(y, M, N, A) := KFD(K)
metric = -log(tr(M)/tr(N))

ans = K %*% y[, 1:2]

if (length(grep("RStudio",  commandArgs(trailingOnly = FALSE))) == 0) {
    png(sprintf("%d_eval.png", epoch), width=1024, height=768)
}
par(mfrow=c(1,2))

metrics_str = sprintf("%f", metric)
plot(Re(ans[,1]), col=as.integer(colnames(K)), main=metrics_str)
plot(Re(ans), col=as.integer(colnames(K)))

cat(metric, "\n")

