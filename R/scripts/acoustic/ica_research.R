require(Rtsne)
require(Rdnn)
require(ica)

spikes = proto.read(spikes.path("impro_eval.pb"))

rv = get.rate.vectors(spikes, 50)
input.signal = t(rv$values)

cl = sapply(rv$info, function(x) x$label)
uc = unique(cl)
rainbow_cols = rainbow(length(uc))

cols = NULL
labs = NULL
tc = 0
for (i in rv$info) {
    col = rainbow_cols[which(i$label == uc)]
    cols = c(cols, rep("black", i$start_time - tc), rep(col, i$duration))
    labs = c(labs, rep("none", i$start_time - tc), rep(i$label, i$duration))
    tc = i$start_time + i$duration
}



for (i in 1:10) {
    ica.signal = icafast(input.signal, 10)
    mixm = abs(ica.signal$M)
    print(gr_pl(mixm))
    
    
    dups = duplicated(ica.signal$S)
    ans.tsne = Rtsne(ica.signal$S[!dups, ])
    cols = cols[!dups]
    plot(ans.tsne$Y, col=cols)
    
    input.signal = ica.signal$S
}



