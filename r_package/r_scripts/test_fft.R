
require(Rdnn)

ts = list(values=sin(0.1*seq(0,1000)))
f="/home/alexeyche/dnn/ts/test.pb"
RProto$new(f)$write(ts, "TimeSeries")

ff="/home/alexeyche/dnn/build/test_fft.pb"
f_inv="/home/alexeyche/dnn/build/test_inv.pb"

f_ts = RProto$new(f)$read()$values[[1]]
ff_ts = RProto$new(ff)$read()$values[[1]]

ff_ts_true = Re(fft(f_ts))/length(f_ts)

f_ts_true = Re(fft(ff_ts_true, inverse=TRUE))
f_ts_false = RProto$new(f_inv)$read()$values[[1]]