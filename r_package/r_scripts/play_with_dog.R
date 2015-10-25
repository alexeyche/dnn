

require(Rdnn)

M=100
L=1000
ts = time.series(matrix(rnorm(L*M), nrow=M, ncol=L))

ts = time.series(
    matrix(c(1:100, rep(0, 50), 100:1), nrow=1)
    , ts.info(c("1","2"), c(150, 250))
)


a=2.0
b=10.0
r=3
start = 1.0

s = RSim$new()
const = s$getConst()

const$setElement("DifferenceOfGaussians", list(a=a, b=b, r=r, dimension=1, apply_amplitude=TRUE))
const$setElement("Sigmoid", list(threshold=1.0, slope=10.0))
const$addLayer(list(
    size=M, neuron="SRMNeuron", act_function="Sigmoid", input="InputTimeSeries"
))
const$addConnection(0, 0, list(
    type="DifferenceOfGaussians"
  , start_weight=start
  , synapse="StaticSynapse"
  , inh_synapse="StaticSynapse_inh"
))

s$build()
s$setTimeSeries(ts, "InputTimeSeries")
s$run(4)
sp = s$getSpikes()

plot(sp)
m = s$getModel()
w = m[[1]]
plotl(w[51,])
