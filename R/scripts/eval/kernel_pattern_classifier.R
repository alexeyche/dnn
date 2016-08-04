#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)

input_neurons = 256
preprocessor = Gauss(TauDecay=10)
kernel = RbfDot(Sigma=0.5)
jobs = 8
dt = 1.0

c(spikes, epoch) := read.spikes.wd()
#epoch = get.last.epoch.wd()
#spikes = proto.read(sprintf("%d_eval_spikes.pb", epoch))

spikes$values = spikes$values[-(1:input_neurons)]

K = pp.class.kernel.run(preprocessor, kernel, spikes, jobs, dt)
#colnames(K) = sample(colnames(K))
c(y, M, N, A) := KFD(K)

metric = -log(tr(M)/tr(N))

ans = K %*% y[, 1:2]

if (length(grep("RStudio",  commandArgs(trailingOnly = FALSE))) == 0) {
    png(sprintf("%d_eval.png", epoch), width=1024, height=768)    
}

#ans = ans[which(colnames(K) == 0), ]
#K = K[which(colnames(K) == 1), which(colnames(K) == 1)]
par(mfrow=c(1,2))

uc = unique(colnames(K))
r = rainbow(length(uc))
cols = r[sapply(colnames(K), function(l) which(l == uc))]
    
metrics_str = sprintf("%f", metric)
plot(Re(ans[,1]), col=cols, main=metrics_str) 
plot(Re(ans), col=cols)        

cat(metric, "\n")