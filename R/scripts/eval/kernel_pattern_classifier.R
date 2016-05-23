#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)

input_neurons = 100
preprocessor = Epsp(TauDecay=10)
kernel = Dot()
jobs = 8


c(spikes, epoch) := read.spikes.wd()

spikes$values = spikes$values[-(1:input_neurons)]

K = pp.class.kernel.run(preprocessor, kernel, spikes, jobs)
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