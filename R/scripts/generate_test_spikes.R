
require(Rdnn)
source(scripts.path("gen_poisson.R"))

N = 100
# 
rates = rnorm(N, mean=10)
sample_duration = 150

inp_spikes = empty.spikes(N)

for(i in 1:N) { # gen pattern
    inp_spikes$values[[i]] = gen_poisson(1, rates[i], sample_duration)[[1]]
}
all_inp_spikes = inp_spikes
for (i in 1:100) {
    all_inp_spikes = add.to.spikes(all_inp_spikes, inp_spikes, gap=100)
}


M = 10
targ_spikes = empty.spikes(M)

#targ_spikes$values[[1]] = c(10, 75)

for (i in 1:M) {
    targ_spikes$values[[i]] = i*(sample_duration/M)
}

all_targ_spikes = targ_spikes
for (i in 1:100) {
    all_targ_spikes = add.to.spikes(all_targ_spikes, targ_spikes, gap=100)
}


proto.write(all_inp_spikes, spikes.path("input_spikes.pb"))
proto.write(all_targ_spikes, spikes.path("target_spikes.pb"))

b_spikes = binarize.spikes(inp_spikes)
gauss = exp(-((seq(0, 1.0, length.out = 50) - 0.5) ** 2/0.01))

# for (ni in 1:nrow(b_spikes$values)) {
#     b_spikes$values[ni,] = filter(b_spikes$values[ni,], gauss, circular=TRUE)
# }
# plot(b_spikes$values[10,], type="l")
proto.write(b_spikes, ts.path("target_spikes.pb"))

