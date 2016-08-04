
require(Rdnn)

N = 15

inp_spikes = vector("list", N)
for (i in 1:N) {
  inp_spikes[[i]] = c(i*5)
}


inp_spikes = spikes.list(inp_spikes)

target_spikes = spikes.list(list(c(37.0)))

proto.write(inp_spikes, spikes.path("input_spikes.pb"))
proto.write(target_spikes, spikes.path("target_spikes.pb"))
