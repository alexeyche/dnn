ltd = function(w, w0, alpha) {
    ans = rep(0, length(w))
    lidx = w <= w0
    ans[lidx] = w[lidx]/w0
    ans[!lidx] = 1.0 + log(1.0 + alpha * (w[!lidx]/w0 - 1.0))/alpha
    
    return(ans)
}

ltp = function(w, w0, beta) {
    exp(-w/(w0*beta))
}

w = seq(0, 1.0, length.out = 100)

w0 = 0.1

plot(w, -ltd(w, w0, 1), type="l", ylim=c(-2.0,2.0), col="blue")
lines(w, ltp(w, w0, 10), col="red")