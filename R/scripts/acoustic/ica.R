require(ica)

#spikes = proto.read(spikes.path("work_licks.pb"))
spikes = proto.read(spikes.path("impro.pb"))
#with_spikes = which(sapply(spikes$values, length) > 0)
#spikes$values = spikes$values[with_spikes]

smooth.spikes = preprocess.run(Epsp(TauDecay=10), binarize.spikes(spikes), 8)
input.signal = t(smooth.spikes$values)
#input.signal = t(preprocess.run(Epsp(TauDecay=10), smooth.spikes, 8)$values)
#test_licks_raw = proto.read(ts.path("test_licks_raw.pb"))
#input.signal = t(test_licks_raw$values)

ica.signal = icafast(input.signal, 10)
mixm = abs(ica.signal$M)
gr_pl(mixm)
write.table(mixm, runs.path("ica_filter.csv"), sep=",", row.names=FALSE, col.names=FALSE)






spikes.le = function(spikes, win=100) {
    cl = sapply(spikes$info, function(x) x$label)
    uc = unique(cl)
    rainbow_cols = rainbow(length(uc))
    
    cols = NULL
    labs = NULL
    rv = get.rate.vectors(spikes, win)
    tc = 0
    for (i in rv$info) {
        col = rainbow_cols[which(i$label == uc)]
        cols = c(cols, rep("black", i$start_time - tc), rep(col, i$duration))
        labs = c(labs, rep("none", i$start_time - tc), rep(i$label, i$duration))
        tc = i$start_time + i$duration
    }
    
    dups = duplicated(t(rv$values))
    ans.tsne = Rtsne(t(rv$values[, !dups]))
    cols = cols[!dups]
    plot(ans.tsne$Y, col=cols)
    return(list(ans.tsne, cols))    
}


