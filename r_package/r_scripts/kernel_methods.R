
normalize = function(K) {
    D = diag(1/sqrt(diag(K)))
    Kn = D %*% K %*% D
    colnames(Kn) <- colnames(K) 
    rownames(Kn) <- rownames(K) 
    return(Kn)
}

centering = function(K) {
    ell = nrow(K)
    D = colMeans(K)/ell
    E = mean(D)/ell
    J = rep(1, ell) %*% t(D)
    K = K - J - t(J) - E * matrix(rep(1, ell*ell), nrow=ell, ncol=ell)
}

 

get.labs = function(K) {
    sapply(
        strsplit(colnames(m), split="[.]"), 
        function(s) tail(s, 1)
    )
}
    