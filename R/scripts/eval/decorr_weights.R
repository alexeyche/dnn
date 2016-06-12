#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)

neurons_of_interest = 257:266
inputs_of_interest = 1:256


epoch = as.numeric(strsplit(system("ls -t *.pb | head -n 1", intern = TRUE), "_")[[1]][1])
model = proto.read(sprintf("%s_model.pb", epoch))

w = matrix(0, nrow=length(model), ncol=length(model))
for(n in model) {
    w[n$id+1, n$synapses$ids_pre+1] = n$synapses$weights
}


mw = matrix(0, nrow=length(inputs_of_interest), ncol=length(neurons_of_interest))

for (ni in 1:length(neurons_of_interest)) {
    mw[, ni] = w[neurons_of_interest[ni], inputs_of_interest]
}

angles = matrix(0, nrow=length(neurons_of_interest), ncol = length(neurons_of_interest))

for (ni in 1:length(neurons_of_interest)) {
    for (nj in 1:length(neurons_of_interest)) {
        if (ni == nj) {
            next
        }
        angles[ni, nj] = t(mw[, ni]) %*% mw[, nj] / (norm(mw[, ni], type="2") * norm(mw[,nj], type="2"))
    }
}

png(sprintf("%d_eval.png", epoch), width=1024, height=768)
print(gr_pl(mw))
dev.off()

cat(mean(angles), "\n")


