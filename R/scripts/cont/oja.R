require(Rdnn)

set.seed(10)

dt = 1.0
M = 1


spikes = proto.read(spikes.path("timed_pattern_spikes.pb"))
with_spikes = which(sapply(spikes$values, length) > 0)
spikes$values = spikes$values[with_spikes]
input.signal = t(preprocess.run(Epsp(TauDecay=10), binarize.spikes(spikes), 8)$values)

N = ncol(input.signal)

W = matrix(runif(N*M), N, M)


#input.signal[abs(input.signal) < 1e-07] <- 0



#input.signal = matrix(rnorm(N*10000), N, 10000)

#input.signal = matrix(rnorm(N*200), 200, N)

K = nrow(input.signal)
eta = 1e-03


neuron = list(
    membrane=rep(0, M), 
    weights=W, 
    tau_mem = 10.0
)

leaky_neuron_calc = function(n, input) {
    n$membrane = n$membrane + dt * ( - n$membrane + t(input %*% n$weights)) / n$tau_mem
    
    return(n)
}

epochs = 10

m.stat = array(dim=c(M, K*epochs))
w.stat = array(dim=c(N, M, K*epochs))
dw.stat = array(dim=c(N, M, K*epochs))

idx.stat = function(ep, i) K * (ep-1) + i

for (ep in 1:epochs) {
    cats("Epoch %s\n", ep)
    
    for (i in 1:K) {
        x = input.signal[i, ]
        neuron = leaky_neuron_calc(neuron, x)    

        dw = x %*% t(neuron$membrane)  - neuron$weights * matrix(rep(neuron$membrane^2, N), N, M, byrow=TRUE)
        
        neuron$weights = neuron$weights + eta * dw
        
        
        m.stat[, idx.stat(ep, i)] = neuron$membrane
        w.stat[,, idx.stat(ep, i)] = neuron$weights
        dw.stat[,, idx.stat(ep, i)] = dw
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
plot(-neuron$weights, type="l")
#lines(-pc$rotation[,1], col="red")

ei = eigen(t(signal) %*% (signal))
lines(Re(ei$vectors[,1]), col="blue")

r.signal = signal %*% Re(ei$vectors[,1])

membr.final = m.stat[, idx.stat(epochs, 1:nrow(input.signal))]

plot(membr.final,type="l")
lines(-r.signal[,1], col="blue")
