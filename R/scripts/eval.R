
require(Rdnn)

fisher_eval = function(eval_spikes, verbose, jobs) {
    K = pp.kernel.run(Epsp(TauDecay=10), Dot(), eval_spikes, 8)
    if(verbose) {
        c(y, M, N, A) := KFD(K)
        return(list(-tr(M)/tr(N), K, y, M, N, A))    
    } else {
        c(M, N, A) := KFD(K, only_scatter=TRUE)        
        return(-tr(M)/tr(N))
    }
    
}