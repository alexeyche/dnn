
require(Rdnn)
# 
# ts = list(values=sin(0.1*seq(0,1000,length.out=1000)))
# f="/home/alexeyche/dnn/ts/test.pb"
# RProto$new(f)$write(ts, "TimeSeries")
# 
# ff="/home/alexeyche/dnn/build/test_fft.pb"
# f_inv="/home/alexeyche/dnn/build/test_inv.pb"
# 
# f_ts = RProto$new(f)$read()$values[[1]]
# ff_ts = RProto$new(ff)$read()$values[[1]]
# 
# ff_ts_true = Re(fft(f_ts))/length(f_ts)
# 
# f_ts_true = Re(fft(ff_ts_true, inverse=TRUE))
# f_ts_false = RProto$new(f_inv)$read()$values[[1]]
# 
tf = seq(0, 100, length.out=100)
f = exp(-tf/20)

set.seed(2)
sp = gen_poisson(1, rate=40, binary=TRUE)[1,]
# f = c(f, rep(0, length(sp)-length(f)))
# 
# f_m_sp = (fft(f) * fft(sp))/length(sp)
# conv = Re(fft(f_m_sp, inverse=TRUE))
# plot(conv[1:200], type="l")
# lines(sp[1:200], type="l", col="red")


file="/home/alexeyche/dnn/build/test.pb"
RProto$new(file)$write(list(values=sp), "TimeSeries")
file="/home/alexeyche/dnn/build/filter.pb"
RProto$new(file)$write(list(values=f), "TimeSeries")

file="/home/alexeyche/dnn/build/out.pb"
out = RProto$new(file)$read()

file="/home/alexeyche/dnn/build/kernel.pb"
kernel = RProto$new(file)$read()

file="/home/alexeyche/dnn/build/gram.pb"
gram = RProto$new(file)$read()
