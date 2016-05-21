require(Rdnn)
require(ica)

set.seed(11)

dt = 1.0
M = 1
tau_pmean = 500
epochs = 10

spikes = proto.read(spikes.path("test_licks.pb"))
with_spikes = which(sapply(spikes$values, length) > 0)
spikes$values = spikes$values[with_spikes]
input.signal = t(preprocess.run(Epsp(TauDecay=10), binarize.spikes(spikes), 8)$values)

N = ncol(input.signal)

W = matrix(-0.1 + 0.2 * runif(N*M), N, M)

K = nrow(input.signal)
eta = 1e-02
num.of.mom = 4


neuron = list(
    weights=W, 
    tau_mem = 10.0,
    y = rep(0, M),
    moments = matrix(0, nrow=M, ncol=num.of.mom),
    moments.stat = vector("list", num.of.mom)
)


act = function(x) {
    1/(1+exp(-x))
}

leaky_neuron_calc = function(n, input) {
    n$y = n$y + dt * ( - n$y + act(t(input) %*% n$weights)) / n$tau_mem
    
    return(n)
}

oja_rule = function(neuron, x, alpha = 1.33) {
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
        
        for (mi in 1:num.of.mom) {
            v = neuron$y^(mi)
            
            neuron$moments[, mi] = neuron$moments[, mi] + (v - neuron$moments[, mi])/tau_pmean 
        
#             if (length(neuron$moments.stat[[mi]]) < tau_pmean) {
#                 neuron$moments.stat[[mi]] = c(neuron$moments.stat[[mi]], v)
#             } else {
#                 if (neuron$moments[, mi] == 0) {
#                     neuron$moments[, mi] = mean(neuron$moments.stat[[mi]])
#                 } else {
#                     neuron$moments[, mi] = neuron$moments[, mi] - neuron$moments.stat[[mi]][1]/tau_pmean  + v/tau_pmean
#                     neuron$moments.stat[[mi]] = c(neuron$moments.stat[[mi]][-1], v)
#                 }
#             }

            mom.stat[, mi, idx.stat(ep, i)] = neuron$moments[, mi]
            mom.inst[, mi, idx.stat(ep, i)] = v
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

par(mfrow=c(3,1))

#signal = filter.signal(input.signal, neuron$tau_mem)

plot(neuron$weights, type="l")

#ei = eigen(cor(signal))
ei = eigen(t(signal) %*% (signal))
lines(Re(ei$vectors[,1]), col="blue")

lines(max(neuron$weights) * sapply(spikes$values,length)/max(sapply(spikes$values,length)), col="red")
ica.signal = icafast(signal, 10)
plot(neuron$weights, type="l")
lines(-ica.signal$M[,1], type="l", col="blue")

r.signal = signal %*% Re(ei$vectors[,1])

y.final = y.stat[, idx.stat(epochs, 1:K)]

plot(y.final,type="l")
lines(act(r.signal[1:K,1]), col="blue")

#plot(-0.333*mom.stat[1,4,10000+1:20000] + 0.25*mom.stat[1,2,10000+1:20000]^2, type="l") # Kurtosis
#mean( (y.stat - mean(y.stat[1,]))^2) / (mean( (y.stat - mean(y.stat[1,]))^2))^2
# require(ica)
# r = icafast(signal, 10)
# plot(r$M[,1],type="l")
# 
# pca_w = rep(NA, length(spikes$values))
# pca_w[which(sapply(spikes$values, length) > 0)] = Re(ei$vectors[,1])
# write.table(pca_w, runs.path("pca_test_licks.csv"), row.names=FALSE, col.names=FALSE, sep=",")
# 
test_licks = proto.read("~/dnn/spikes/test_licks.pb")

test_licks = add.ts.info(test_licks, ts.info(label="0", start_time=150, duration=500))
test_licks = add.ts.info(test_licks, ts.info(label="1", start_time=740, duration=360))
test_licks = add.ts.info(test_licks, ts.info(label="2", start_time=1250, duration=550))
test_licks = add.ts.info(test_licks, ts.info(label="3", start_time=1850, duration=650))
test_licks = add.ts.info(test_licks, ts.info(label="4", start_time=2550, duration=250))
test_licks = add.ts.info(test_licks, ts.info(label="5", start_time=2800, duration=350))
test_licks = add.ts.info(test_licks, ts.info(label="6", start_time=3200, duration=900))
test_licks = add.ts.info(test_licks, ts.info(label="7", start_time=4150, duration=520))
test_licks = add.ts.info(test_licks, ts.info(label="8", start_time=4670, duration=630))
test_licks = add.ts.info(test_licks, ts.info(label="9", start_time=5300, duration=900))

while (TRUE) {
    max_t = max(sapply(test_licks$values, function(x) if(length(x)>0) { max(x)} else {0}))
    if (max_t > 30000) {
        break
    }
    test_licks = add.to.spikes(test_licks, test_licks)
}

proto.write(test_licks, "/home/alexeyche/dnn/spikes/work_licks.pb")
