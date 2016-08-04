#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)
require(gridExtra)

set.verbose.level(1)

neurons_of_interest = 16:16

c(spikes, epoch) := read.spikes.wd()

spikes$values = spikes$values[neurons_of_interest]

target_sequence = proto.read(spikes.path("target_spikes.pb"))

#preprocessor = Gauss(Sigma=1.0, Length=5, Dt=1.0)
preprocessor = Epsp(TauDecay=10)
kernel = RbfDot(Sigma=0.1)
#kernel = Dot()

#spikes$values = list(c(10.0, 15.0, 25.0))
#target_sequence$values = list(c(11.0, 15.0, 25.0))

#spikes.pp = preprocess.run(preprocessor, binarize.spikes(spikes), 8)
#plot(c(spikes.pp$values))

metric = pp.spike.distance(preprocessor, kernel, spikes, target_sequence, 8)

if (length(grep("RStudio",  commandArgs(trailingOnly = FALSE))) == 0) {
  png(sprintf("%d_eval.png", epoch), width=1024, height=768)    
}
if(sum(sapply(spikes$values, length))>0) {
  grid.arrange(plot(spikes, T0=0, Tmax=70, main=sprintf("%.3f", metric)), plot(target_sequence, T0=0, Tmax=70), nrow=2)
  
}

cat(-metric, "\n")