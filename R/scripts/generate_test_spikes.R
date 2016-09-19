
require(Rdnn)
source(scripts.path("gen_poisson.R"))

N = 25

rates = rnorm(N, mean=25)
sample_duration = 1000
inp_spikes = empty.spikes(N)

for(i in 1:N) { # gen pattern
    inp_spikes$values[[i]] = N*i #gen_poisson(1, rates[i], sample_duration)[[1]]
}

proto.write(inp_spikes, spikes.path("input_spikes.pb"))
proto.write(inp_spikes, spikes.path("target_spikes.pb"))

b_spikes = binarize.spikes(inp_spikes)
gauss = exp(-((seq(0, 1.0, length.out = 50) - 0.5) ** 2/0.01))

# for (ni in 1:nrow(b_spikes$values)) {
#     b_spikes$values[ni,] = filter(b_spikes$values[ni,], gauss, circular=TRUE)
# }
plot(b_spikes$values[10,], type="l")
proto.write(b_spikes, ts.path("target_spikes.pb"))

