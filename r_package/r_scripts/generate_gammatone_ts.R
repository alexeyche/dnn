

sel=c(1:5, 51:55, 101:105, 151:155, 201:205, 251:255)

c(train_ts, test_ts) := prepare.ucr.data(60, UCR.SYNTH, gap_between_patterns = 150, sel=sel)


M = 100
low_f = 10
high_f = 300
samp_rate = 1000
seq.fun = log.seq

data_conv = conv.gammatones(train_ts, seq.fun(low_f, high_f, length.out=M), samp_rate)
proto.write(data_conv, ts.path("ts_gammatones.pb"))
