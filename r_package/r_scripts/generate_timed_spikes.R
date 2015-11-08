
require(Rdnn)
source(scripts.path("gen_poisson.R"))

sim_length = 60000
neurons = 100
sample_gap = 100
sample_duration = 500
classes = 2 
high_rate = 10

patterns_spikes = empty.spikes(neurons)
for(cl in 1:classes) {
    spikes = empty.spikes(neurons)
    rates = high_rate*rbeta(neurons, 0.2,0.9)
    for(i in 1:neurons) {
        spikes$values[[i]] = gen_poisson(1, rates[i], sample_duration)[[1]]
    }
    spikes$ts_info$labels_timeline = c(spikes$ts_info$labels_timeline, sample_gap + sample_duration)
    spikes$ts_info$labels_ids = c(0)
    spikes$ts_info$unique_labels = as.character(cl)
    patterns_spikes = cat.spikes(patterns_spikes, spikes)
}

final_spikes = empty.spikes(neurons)
while(TRUE) {
    final_spikes = cat.spikes(final_spikes, patterns_spikes)
    if(tail(final_spikes$ts_info$labels_timeline, n=1)>sim_length) {
        break
    }
}
proto.write(final_spikes, spikes.path("timed_pattern_spikes.pb"))
    
