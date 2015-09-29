
require(Rdnn)

tr = function(m) {
    sum(diag(m))
}

KFD = function(K, only_ratio=FALSE, mu=0.001) {
    y = colnames(K)
    cls = unique(y)
    
    N = matrix(0, nrow = length(y), ncol = length(y))
    M = matrix(0, nrow = length(y), ncol = length(y))
    Ms = NULL
    for(lab in cls) {
        l_idx = which(y == lab)
        l = length(l_idx)
        
        Mi = rowSums(K[, l_idx])/l
        Ki = K[,l_idx]
        eye = diag(l)
        eye_l = matrix(1/l, nrow=l, ncol=l)
        N = N + Ki %*% (eye - eye_l) %*% t(Ki)
        Ms = cbind(Ms, Mi)
    }
    
    M0 = rowSums(Ms)/length(cls)
    for(li in 1:length(cls)) {
        M = M + (Ms[, li] - M0) %*% t(Ms[, li] - M0)
    }
    
    A = solve(N+mu*diag(length(y))) %*% M
    if(only_ratio) {
        return(list(M=M, N=N, A=A))
    }
    Ae = eigen(A)
    return(list(y=Ae$vectors, M=M, N=N, A=A))
}


require(MASS)

par(mfrow=c(2,2))
Nsamples = 100


data = NULL
data = rbind(data, mvrnorm(
    Nsamples
  , Sigma=diag(2)
  , mu=c(2,2)
))
data = rbind(data, mvrnorm(
    Nsamples
    , Sigma=diag(2)
    , mu=c(0,3)
))
data = rbind(data, mvrnorm(
    Nsamples
    , Sigma=diag(2)
    , mu=c(2,5)
))
rownames(data) <- c(rep("1", Nsamples), rep("2", Nsamples), rep("3", Nsamples))

data = data[ sample(nrow(data)), ]

d = lda(x=data, grouping=as.factor(rownames(data)))
dv = predict(d, data)

K = data %*% t(data)
c(y, M, N, A) := KFD(K)


metrics_str = sprintf("Metrics: %f, %f", tr(M)/tr(N), tr(A))

ans = K %*% y[, 1:2]

plot(data, col=as.integer(rownames(data)))
plot(dv$x[,1], col=as.integer(rownames(data)))
plot(Re(ans[,1]), col=as.integer(rownames(data)), main=metrics_str) # eurica!
plot(Re(ans), col=as.integer(rownames(data))) # eurica!
