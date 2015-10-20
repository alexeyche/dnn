

require(Rdnn)

f = "/home/alexeyche/dnn/runs/test-run/1_spikes.pb"

sp = proto.read(f)

neurons = 101:200

sp$values = sp$values[neurons]


ts = time.series(
    matrix(c(1:100, 100:1), nrow=1)
    , ts.info(c("1","2"), c(100, 200))
)
