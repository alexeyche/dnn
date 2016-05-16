#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)

epoch = as.numeric(strsplit(system("ls -t *.pb | head -n 1", intern = TRUE), "_")[[1]][1])
model = proto.read(sprintf("%s_model.pb", epoch))
w = model[[257]]$synapses$weights

pca_w = unlist(read.table(runs.path("pca_test_licks.csv"), sep=","))
pca_w_val = !sapply(pca_w, is.na)
metric = 1000*sqrt(mean((pca_w[pca_w_val] - w[pca_w_val])^2))
cat(metric, "\n") 

