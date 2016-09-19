#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)
require(gridExtra)

set.verbose.level(1)

neurons_of_interest = 36:(35+25)

c(spikes, epoch) := read.spikes.wd()

spikes$values = spikes$values[neurons_of_interest]

target_sequence = proto.read(spikes.path("target_spikes.pb"))

#preprocessor = Gauss(Sigma=1.0, Length=5, Dt=1.0)
preprocessor = Epsp(TauDecay=10)
#kernel = RbfDot(Sigma=0.1)
kernel = Dot()

#spikes$values = list(c(10.0, 15.0, 25.0))
#target_sequence$values = list(c(11.0, 15.0, 25.0))

#spikes.pp = preprocess.run(preprocessor, binarize.spikes(spikes), 8)
#plot(c(spikes.pp$values))

spikes.pp = binarize.spikes(spikes)
spikes.pp$values = cbind(spikes.pp$values, matrix(0, nrow=nrow(spikes.pp$values), ncol=50))
spikes.pp = preprocess.run(preprocessor, spikes.pp, 8)

target_sequence.pp = binarize.spikes(target_sequence)
target_sequence.pp$values = cbind(target_sequence.pp$values, matrix(0, nrow=nrow(target_sequence.pp$values), ncol=50))
target_sequence.pp = preprocess.run(preprocessor, target_sequence.pp, 8)

difflen = ncol(spikes.pp$values) - ncol(target_sequence.pp$values)

if (difflen > 0) {
    target_sequence.pp$values = cbind(target_sequence.pp$values, matrix(0, nrow=nrow(target_sequence.pp$values), ncol=difflen))
} else
if (difflen < 0) {
    spikes.pp$values = cbind(spikes.pp$values, matrix(0, nrow=nrow(spikes.pp$values), ncol=abs(difflen)))
}

metric = sum((spikes.pp$values - target_sequence.pp$values)^2) 

if (length(grep("RStudio",  commandArgs(trailingOnly = FALSE))) == 0) {
  png(sprintf("%d_eval.png", epoch), width=1024, height=768)    
}
if(sum(sapply(spikes$values, length))>0) {
  grid.arrange(plot(spikes, main=sprintf("%.3f", metric)), plot(target_sequence), nrow=2)
}

cat(metric, "\n")