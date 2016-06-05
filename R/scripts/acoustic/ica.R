require(ica)

spikes = proto.read(spikes.path("test_licks.pb"))
#with_spikes = which(sapply(spikes$values, length) > 0)
#spikes$values = spikes$values[with_spikes]

smooth.spikes = preprocess.run(Epsp(TauDecay=10), binarize.spikes(spikes), 8)
input.signal = t(smooth.spikes$values)
input.signal = t(preprocess.run(Epsp(TauDecay=10), smooth.spikes, 8)$values)
#test_licks_raw = proto.read(ts.path("test_licks_raw.pb"))
#input.signal = t(test_licks_raw$values)

ica.signal = icafast(input.signal, 10)
mixm = abs(ica.signal$M)
gr_pl(mixm)
write.table(mixm, runs.path("ica_filter.csv"), sep=",", row.names=FALSE, col.names=FALSE)


