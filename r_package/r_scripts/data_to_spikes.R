

require(Rdnn)

sel=c(1:10, 50:60, 100:110, 150:160, 200:210, 250:260)
c(train_ts, test_ts) := prepare.ucr.data(1000, UCR.SYNTH, gap_between_patterns = 0, sel=sel)

N=100
dt=1


sp = intercept.data.to.spikes(
    train_ts
  , N
  , 1
  , dt
  , 100
)

proto.write(sp, spikes.path("simple_spikes.pb"))


