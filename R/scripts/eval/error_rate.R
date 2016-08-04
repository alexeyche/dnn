#!/usr/bin/env Rscript

require(methods, quietly=TRUE)
require(Rdnn, quietly=TRUE)

# error rate, density of probability and ROC-curve (required package "ROCR")
# conditions:
#   - last layer is output InputClassifier layer
#   - quantity of neurons in last layer is equivalent to quantity of classes
#   - min quantity of classes is 2

epoch=as.numeric(strsplit(system(sprintf("basename $(ls -t %s/*.pb | head -n 1)", getwd()), intern=TRUE), "_")[[1]][1])
spikes = proto.read(sprintf("%d_spikes.pb", epoch))
print.roc = FALSE
if ("ROCR" %in% installed.packages()[,"Package"]) {
  require("ROCR")
  print.roc = TRUE
}
chop.spikes = chop.spikes.list(spikes)
errors = c()
errors.count = 0
probability.list = list()
auc.vec = c()
labels.vec = c()

for (i in 1: length(spikes$info)) {
  labels.vec = c(labels.vec, spikes$info[[i]]$label)
}
labels.vec = unique(labels.vec)

last.neuron = length(spikes[[1]])
first.neuron = last.neuron - length(labels.vec)

for (i in 1:(last.neuron - first.neuron)) {
  probability.list[[i]] = matrix(NA, 0, length(labels.vec))
}

for(i in 1:length(chop.spikes)) {
  activity.vec = c()  # neurons activity vector
  quantity.vec = c()  # quantity of spikes vector
  
  for (j in (first.neuron + 1):last.neuron) {
    activity.vec = c(activity.vec, length(chop.spikes[[i]]$values[[j]])/chop.spikes[[i]]$info[[1]]$duration)
    quantity.vec = c(quantity.vec, length(chop.spikes[[i]]$values[[j]]))
  }
  
  # errors count
  if (max(activity.vec) == 0
      | max(table(activity.vec)) > 1
      | chop.spikes[[i]]$info[[1]]$label != labels.vec[which.max(activity.vec)] ) {
    errors.count = errors.count + 1
  }
  errors = c(errors, errors.count)
  
  # probability
  for (j in 1:length(quantity.vec)) {
    row = c()
    if (chop.spikes[[i]]$info[[1]]$label == labels.vec[j] ) {
      for (k in 1:length(quantity.vec)) {
        sum = sum(quantity.vec)
        if (sum != 0) {
          row = c(row, quantity.vec[[k]]/sum)
        } else {
          row = c(row, sum)
        }
      }
      probability.list[[j]] = rbind(probability.list[[j]], row)
    }
  }
}
# plots
if (length(grep("RStudio",  commandArgs(trailingOnly = FALSE))) == 0) {
  png(sprintf("%d_eval.png", epoch), width=1024, height=768)    
}
if (print.roc) {
  par(mfrow = c((last.neuron - first.neuron), (last.neuron - first.neuron)))
} else {
  par(mfrow = c((last.neuron - first.neuron - 1), (last.neuron - first.neuron - 1)))
}
# errors rate in current epoch
error.rate = errors[[length(errors)]]/length(errors)
plot(errors, main = sprintf("Epoch #%d, error rate %.3f", epoch, error.rate),
     ylab = "Number of errors", type = "s", col = "darkred", lwd = 2)

for (n in 1:(last.neuron - first.neuron)) {
  if (n != (last.neuron - first.neuron)) {
    for (m in (n + 1):(last.neuron - first.neuron)) {
      y.max = max(c(max(density(probability.list[[m]][, m])$y), max(density(probability.list[[n]][, n])$y)))
      plot(c(0,y.max), type = "n", ylab = "Density", xlab = "Probability",
           main = sprintf("Densities, classes %s and %s", labels.vec[n], labels.vec[m]),
           xlim = 0:1, xaxt = 'n', yaxt = 'n')
      axis(1, at=c(0, 0.25, 0.5, 0.75, 1))
      y.axis.max = y.max/length(probability.list[[m]][, m])
      axis(2, at=c(0, y.max/4, y.max/2, 3*y.max/4, y.max), labels = c(0, round(y.axis.max/4, 2), round(y.axis.max/2, 2), round(3*y.axis.max/4, 2), round(y.axis.max, 2)))
      grid(nx = 2, ny = NA, lty = 1, lwd = 2)
      lines(density(probability.list[[n]][, n]), xlim = 0:1, lty = 1, col = n, lwd = 2)
      lines(density(1 - probability.list[[m]][, m]), xlim = 0:1, lty = 1, col = m, lwd = 1.5)
      legend("topleft", bty = "n", c(labels.vec[n], labels.vec[m]),
             lty = c(1, 1), lwd = c(2, 1.5), col = c(n, m))
      # ROC-curve and AUC evaluation
      if (print.roc) {
        tmp.n = length(probability.list[[n]][, n])
        tmp.m = length(probability.list[[m]][, m])
        performances = performance(prediction(c(1 - probability.list[[n]][, n], probability.list[[m]][, m]),
                                              c(rep(labels.vec[n], tmp.n), rep(labels.vec[m], tmp.m))), "tpr", "fpr")
        plot(performances, avg = "threshold", colorize = T, lwd = 4,
             main = sprintf("ROC-curve, classes %s and %s", labels.vec[n], labels.vec[m]))
        auc = performance(prediction(c(1 - probability.list[[n]][, n], probability.list[[m]][, m]),
                                     c(rep(labels.vec[n], tmp.n), rep(labels.vec[m], tmp.m))), "auc")
        legend("bottomright", bty = "n", sprintf("AUC = %.3f  ", auc@y.values), lty = 0)
        auc.vec = c(auc.vec, auc@y.values[[1]])
      }
    }
  }
}
if (print.roc) {
  cat(sprintf("%1.10f", 0.25 * error.rate + 0.75 * (1 - mean(auc.vec))), "\n")
} else {
  cat(sprintf("%1.10f", error.rate), "\n")
}
