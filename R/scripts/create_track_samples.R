#!/usr/bin/env Rscript
require(Rdnn)

matrix1 <- t(as.matrix(read.table("matrix_good.txt")))
matrix2 <- t(as.matrix(read.table("matrix_bad.txt")))
info <- list()
samples <- matrix(NA, nrow = 9, ncol = 0)

for (i in 1:8) {
  #good.sample <- matrix1[,((1000 * (i - 1) + 1):(1000 * i))]
  #bad.sample <- matrix2[,((1000 * (i - 1) + 1):(1000 * i))]
  info = c(info, list(
    list(label="1", start_time = (i - 1) * 1000, duration = 500),
    list(label="2", start_time = (i * 2 - 1)* 500, duration = 500)))
  samples <- cbind(samples, matrix1[,((500 * (i - 1) + 1):(500 * i))], matrix2[,((500 * (i - 1) + 1):(500 * i))])
}

proto.write(time.series(samples, info), ts.path("test_ts.pb"))
