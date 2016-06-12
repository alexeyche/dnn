require(Rdnn)
center.corr = function(K) {
    ell = nrow(K)
    D = colSums(K)/ell
    E = sum(D)/ell
    J = rep(1, ell) %*% t(D)
    K - J - t(J) + E * matrix(1, ell, ell)
}
spectral.clust = function(A) {
    D = diag(colSums(A))
    Lsimple = D - A
    Lnorm = sqrt(D) %*% Lsimple %*% sqrt(D)
    eigen(Lnorm)
}

input_neurons = 256

c(spikes, epoch) := read.spikes.wd()
sp = spikes
sp$values = sp$values[-(1:input_neurons)]



samples = chop.spikes.list(sp)
rates = t(sapply(samples, function(x) sapply(x$values, function(sp) length(sp)/x$info[[1]]$duration)))

cl = sapply(samples, function(x) x$info[[1]]$label)
uc = unique(cl)
rainbow_cols = rainbow(length(uc))
cols = rainbow_cols[sapply(cl, function(l) which(l == uc))]


fit <- kmeans(rates, 2)

library(cluster) 
clusplot(rates, fit$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)


plot(svd(rates)$u[,1:2], col=cols)
plot(hclust(dist(rates)))

K = pp.class.kernel.run(Epsp(TauDecay=15), Dot(), sp, 8, 1.0)

plot(hclust(as.dist(K)))

ei = eigen(center.corr(K))
plot(ei$vectors[,1:2], col=cols)


cols = NULL
rv = get.rate.vectors(sp, 50)
tc = 0
for (i in rv$info) {
    col = rainbow_cols[which(i$label == uc)]
    cols = c(cols, rep(0, i$start_time - tc), rep(col, i$duration))
    tc = i$start_time + i$duration
}

plot(svd(t(rv$values))$u[,1:2], col=cols)
plot(hclust(dist(rates)))

