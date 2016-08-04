#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)

norm = function(w, p=2.0) {
    w/(sum(w^p)^(1.0/p))
}


epoch = as.numeric(strsplit(system("ls -t *.pb | head -n 1", intern = TRUE), "_")[[1]][1])
model = proto.read(sprintf("%s_model.pb", epoch))

w = matrix(0, nrow=length(model), ncol=length(model))
for(n in model) {
    w[n$id+1, n$synapses$ids_pre+1] = n$synapses$weights
}

ica.signal = as.matrix(read.table(runs.path("ica_filter.csv"), sep=","))

w = t(w[257:nrow(w), 1:256])

if (any(is.na(w))) {
    stop("Got NA")
}
w.sort = matrix(0, nrow=nrow(w), ncol = ncol(w))
i.sort = matrix(0, nrow=nrow(w), ncol = ncol(w))

# for (si in 1:ncol(w)) {
#     w[,si] = norm(w[,si])
#     ica.signal[,si] = norm(abs(ica.signal[,si]))
# }

dd = t(abs(ica.signal)) %*% w
used_comp = 1:ncol(ica.signal)
for (di in 1:nrow(dd)) {
    win = which(max(dd[di, used_comp]) == dd[di,used_comp])
    dm = used_comp[win]
    w.sort[, dm] = w[, dm]
    i.sort[, dm] = ica.signal[, dm]
    used_comp = used_comp[-win]
}

metric = sqrt(sum((w.sort - i.sort)^2))
cat(metric, "\n") 

