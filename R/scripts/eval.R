
require(Rdnn)

overlap_eval = function(eval_spikes, const) {
    spike_patterns = chop.spikes.list(eval_spikes)
    labs = sapply(spike_patterns, function(x) x$info[[1]]$label)
    ulabs = unique(labs)
    
    con_types = sapply(const$sim_configuration$conn_map, function(x) sapply(x, function(y) y$type))
    dog_idx = grep("DifferenceOfGaussians", con_types)
    if(length(dog_idx) > 0) {
        dim = unique(sapply(const$connections[con_types[dog_idx] ], function(x) x$dimension))
        if(length(dim)>1) {
            warning("Got a lot of DifferenceOfGaussians dimensions in setup. Choosing first for evaluating")
            dim = dim[1]
        }        
    }
    
    N = length(spike_patterns[[1]]$values)
    vv = vector("list", length(ulabs))
    for(el_i in 1:length(labs)) {
        l = labs[el_i]
        li = which(l == ulabs)
        vv[[li]] = rbind(
            vv[[li]],
            sapply(spike_patterns[[el_i]]$values, length)/spike_patterns[[el_i]]$info[[1]]$duration
        )
    }
    vm = sapply(vv, colMeans)
    
    norm = function(x) sqrt(sum(x^2))
    rect = function(x) { x[x<0] <-0; x }
    # vm = t(matrix(c(rep(1, 10), rep(0, 10), rep(0,15), rep(1,5)), nrow=2, ncol=20, byrow=TRUE))
    
    Kc = matrix(0, nrow=ncol(vm), ncol=ncol(vm))
    
    for(li in 1:ncol(vm)) {
        for(lj in 1:ncol(vm)) {
            v1 = vm[,li]
            v2 = vm[,lj]
            
            v1n = norm(v1)
            v2n = norm(v2)
            
            if(v1n>0) v1 = v1/v1n
            if(v2n>0) v2 = v2/v2n
            
            Kc[li, lj] = norm(rect(v1 - v2))            
        }
    }
    metric = sum(Kc)/(nrow(Kc)*nrow(Kc) - nrow(Kc))
    
    #max_rate = 20
    #f = function(p) - (p/max_rate)*log2(p/max_rate) - (1-(p/max_rate))*log2(1-(p/max_rate) )
    #rate = 1000*mean(vm)
    metric = 1-metric
    return(list(metric, vm))
}

fisher_eval = function(eval_spikes, verbose) {
    chopped = chop.spikes.list(eval_spikes)
    K = kernel.run(eval_spikes, EVAL_PROC, EVAL_KERN, jobs=EVAL_JOBS)
    if(verbose) {
        c(y, M, N, A) := KFD(K)
        return(list(-tr(M)/tr(N), K, y, M, N, A))    
    } else {
        c(M, N, A) := KFD(K, only_scatter=TRUE)        
        return(-tr(M)/tr(N))
    }
    
}