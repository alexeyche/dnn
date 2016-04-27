#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)
require(Rtsne, quietly=TRUE)

#input_neurons = 100
neurons_of_interest = 501:600
L = 100

c(spikes, epoch) := read.spikes.wd()
ts = read.input.ts.wd()

spikes$values = spikes$values[neurons_of_interest]
#spikes$values = spikes$values[-(1:input_neurons)]

chopped = chop.spikes.list(spikes)

ensemble = vector("list", length(chopped[[1]]$values))
for (sp in chopped) {
    for (ni in 1:length(sp$values)) {
        for (tsp in sp$values[[ni]]) {
            lb = tsp - L + 1
            ub = tsp
            if (lb < 1) {
                lb = 1
            }
            trigg = ts$values[1, lb:ub]
            if (length(trigg) < L) {
                trigg = c(rep(0, L-length(trigg)), trigg)
            }
            ensemble[[ni]] = rbind(ensemble[[ni]], trigg)
        }
    }    
}

stat = list()

for (ens in ensemble) {
    rownames(ens) <- NULL
    
    sta = colMeans(ens)
    hc = hclust(dist(ens))
    ct = cutree(hc, h=15)
    
    c.means = NULL
    c.var = NULL
    c.elems = list()
    for (k in unique(ct)) {
        m = ens[k == ct,]
        if (is.null(nrow(m))) {
            m = as.matrix(m, nrow=1, ncol=length(m))
        }
        cm = colMeans(m)
        c.means = rbind(c.means, cm)
        v = colMeans((m - rep(1, nrow(m)) %*% t(cm)) ^ 2)
        c.var = rbind(c.var, v)
        c.elems = add.to.list(c.elems, m)
    }
    
    stat = add.to.list(stat, list(
        c.means = c.means,
        c.var = c.var,
        c.elems = c.elems,
        ct = ct,
        sta = sta
    ))
}

plot(sapply(stat, function(x) mean(x$c.var)),type="l")
par(mfrow=c(3, 1))

n = 10
cstat = stat[[n]]
c.elems = cstat$c.elems
c.means = cstat$c.means
c.var = cstat$c.var
ct = cstat$ct
sta = cstat$sta

ens = ensemble[[n]]

plot(sta, type="l", lwd=5, ylim=c(0.05, max(sapply(c.elems, max))), main=c.var.str)
for (k in 1:nrow(c.means)) {
    m = c.elems[[k]]
    for (ri in 1:nrow(m)) {
        lines(m[ri,], col=1, lwd=0.5, type="l")
    }
    lines(c.means[k,], col=k+1, lwd=5, type="l")
}

ei = eigen(t(ens) %*% ens)
par(new=TRUE)
plot(ei$values, xaxt="n",yaxt="n", xlab="",ylab="", col="blue")
axis(4)
mtext("y2",side=4,line=3)

pc = prcomp(ens)
for (k in unique(ct)) {
    if (k > 1) par(new=TRUE)
    plot(pc$x[k == ct,1:2], col=k+1, xlab="", ylab="", xaxt="n",yaxt="n")  
}
plot(hc)
