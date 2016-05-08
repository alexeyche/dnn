require(Rdnn)

set.seed(10)

dt = 1.0
M = 1
tau_pmean = 1000.0
epochs = 10

spikes = proto.read(spikes.path("timed_pattern_spikes.pb"))
with_spikes = which(sapply(spikes$values, length) > 0)
spikes$values = spikes$values[with_spikes]
input.signal = t(preprocess.run(Epsp(TauDecay=10), binarize.spikes(spikes), 8)$values)

N = ncol(input.signal)

W = matrix(1e-01*runif(N*M), N, M)

K = nrow(input.signal)
eta = 1e-04
num.of.mom = 2


neuron = list(
    weights=W, 
    tau_mem = 10.0,
    y = rep(0, M),
    moments = matrix(0, nrow=M, ncol=num.of.mom)
)


act = function(x) {
    1/(1+exp(-x))
}

leaky_neuron_calc = function(n, input) {
    n$y = n$y + dt * ( - n$y + act(t(input) %*% n$weights)) / n$tau_mem
    
    return(n)
}

oja_rule = function(neuron, x, alpha = 2.0) {
    x %*% neuron$y  - alpha*neuron$weights * matrix(rep(neuron$y^2, N), N, M, byrow=TRUE)
}

bcm_rule = function(neuron, x) {
    act_deriv = matrix(rep(neuron$y * (1 - neuron$y), N), nrow=N, ncol=M, byrow=TRUE)
    t((neuron$y * (neuron$y - neuron$moments[, 2])) %*% t(x)) * act_deriv
}

norm = function(w, p=2.0) {
    w/(sum(w^p)^(1.0/p))
}

m.stat = array(dim=c(M, K*epochs))
w.stat = array(dim=c(N, M, K*epochs))
dw.stat = array(dim=c(N, M, K*epochs))
y.stat = array(dim=c(M, K*epochs))
mom.stat = array(dim=c(M, num.of.mom, K*epochs))
mom.inst = array(dim=c(M, num.of.mom, K*epochs))

idx.stat = function(ep, i) K * (ep-1) + i


for (ep in 1:epochs) {
    cats("Epoch %s\n", ep)
    
    for (i in 1:K) {
        x = as.matrix(input.signal[i, ])
        neuron = leaky_neuron_calc(neuron, x)    

        #dw = oja_rule(neuron, x)
        dw = bcm_rule(neuron, x)
        if (ep == 1) {
            dw = 0 # To collect stat
        }
        neuron$weights = neuron$weights + eta * dw  
        neuron$weights = norm(neuron$weights, p = 2.0)
        
        y.stat[, idx.stat(ep, i)] = neuron$y
        w.stat[,, idx.stat(ep, i)] = neuron$weights
        dw.stat[,, idx.stat(ep, i)] = dw
        for (ni in 1:M) {
            for (mi in 1:num.of.mom) {
                neuron$moments[ni, mi] = neuron$moments[ni, mi] + (-neuron$moments[ni, mi] + neuron$y^(mi))/tau_pmean 
                mom.inst[ni, mi, idx.stat(ep, i)] = neuron$y^(mi)
            }
            mom.stat[ni,, idx.stat(ep, i)] = neuron$moments[ni,]
        }
    }
}

filter.signal = function(signal, tau) {
    filtered = matrix(rep(0, ncol(signal)), nrow=1, ncol = ncol(signal))
    for (i in 1:nrow(signal)) {
        filtered = rbind(filtered, filtered[i,] + dt * (-filtered[i,] + signal[i,]) / tau)
    }
    return(filtered[2:nrow(filtered), ])
}

par(mfrow=c(2,1))

#signal = filter.signal(input.signal, neuron$tau_mem)
#signal = input.signal

pc = prcomp(signal, scale=TRUE)
plot(neuron$weights, type="l")
#lines(-pc$rotation[,1], col="red")

ei = eigen(t(signal) %*% (signal))
lines(-Re(ei$vectors[,1]), col="blue")

r.signal = signal %*% Re(ei$vectors[,1])

y.final = y.stat[, idx.stat(epochs, 1:5000)]

plot(y.final,type="l")
lines(act(-r.signal[1:5000,1]), col="blue")

#plot(mom.stat[1,4,1:10000] - 3*mom.stat[1,2,1:10000]^2, type="l") # Kurtosis
#mean( (y.stat - mean(y.stat[1,]))^2) / (mean( (y.stat - mean(y.stat[1,]))^2))^2

