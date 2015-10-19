
require(Rdnn)

s = RSim$new()

const = s$getConst()
const$setElement("Sigmoid", list(threshold=1.0, slope=5.0))
#const$setElement("ExpThreshold", list(threshold=0.70, beta=20.0, amp=20.0, p_rest=0.1))
const$addLayer(
    list(size=100, neuron="LeakyIntegrateAndFire", act_function="Sigmoid", input="InputTimeSeries")
)
const$setNeuronsToListen(c(0))

s$build()
ts = time.series(rnorm(1000, mean=0.0),ts.info("1",1000))
s$setTimeSeries(ts, "InputTimeSeries")

s$turnOnStatistics()
s$run(1)
spikes = s$getSpikes()

stat = s$getStat()

plot(stat[[1]][["LeakyIntegrateAndFire_u"]], type="l")

