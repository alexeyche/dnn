require(onlinePCA)
require(cluster)


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

get.elem = function(data, i) {
    patt.inf = data$info[[i]]
    data$values[,patt.inf$start_time:(patt.inf$start_time + patt.inf$duration-1)]
}

time.corr.pca = function(data, T=1000, q=5) {
    pca = prcomp(data$values[,1:T])
    xbar = pca$center
    pca <- list(values=pca$sdev[1:q]^2, vectors=pca$rotation[,1:q])
    sampSize  = T
    for (i in seq(T+1, ncol(spw.p$values), by=T)) {
        if ( (i+T) > ncol(spw.p$values)) {
            break
        }
        for (j in 1:nrow(data$values)) {
            x = data$values[j, i:(i+T-1)]
            xbar = updateMean(xbar, x, sampSize)
            pca = incRpca(pca$values, pca$vectors, x, sampSize, q=q, center=xbar)
            sampSize = sampSize + 1
        }
    }
    return(pca)
}
no.info = function(ts) {
    ts$info = list()
    return(ts)
}

set.verbose.level(1)

spw = proto.read(sprintf("%d_spikes.pb", EP))
spw$values = spw$values[257:length(spikes$values)]

with_spikes = which(sapply(spw$values, length) > 0)
spw$values = spw$values[with_spikes]

spw.p = preprocess.run(Gauss(Sigma=10.0), binarize.spikes(spw), 8)
time.corr=FALSE

Klist = kernel.run(Dot(), no.info(spw.p), 8, FALSE)

d = get.elem(spw.p, 4)

r = svd(spw.p$values)
sp =r$u[1:4,] %*% d
plot(sp[1,],type="l", col="red", ylim=c(min(sp), max(sp)))
lines(sp[2,], col="blue")

K = center.corr(Klist[[1]])

ans = eigen(K)
V = ans$vectors
L = ans$values
le = ans$vectors[,1:2]

if (time.corr) {
    d.le = t(le) %*% t(d)
    
} else {
    d.le = t(le) %*% d
    #plot(d.le[1,], d.le[2,], type="l")
    plot(d.le[1,],type="l", col="red", ylim=c(min(d.le), max(d.le)))
    lines(d.le[2,],type="l", col="blue")
}

#tc.le = time.corr.pca(spw.p, 1000, 5)
#tc.d.le = t(tc.le$vectors[1:ncol(d),]) %*% t(d)

#plot(tc.d.le[1,],type="l", col="red", ylim=c(min(tc.d.le), max(tc.d.le)))
#lines(tc.d.le[2,],type="l", col="blue")

