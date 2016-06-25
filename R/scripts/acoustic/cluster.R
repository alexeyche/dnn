require(Rtsne)

input_neurons = 256

c(spikes, epoch) := read.spikes.wd()
sp = spikes
sp$values = sp$values[-(1:input_neurons)]

win = 25
rv = get.rate.vectors(sp, win)


cl = sapply(sp$info, function(x) x$label)
uc = unique(cl)
rainbow_cols = rainbow(length(uc))

cols = NULL
labs = NULL
tc = 0
for (i in rv$info) {
    col = rainbow_cols[which(i$label == uc)]
    cols = c(cols, rep("black", i$start_time - tc), rep(col, i$duration))
    labs = c(labs, rep("none", i$start_time - tc), rep(i$label, i$duration))
    tc = i$start_time + i$duration
}


dups = duplicated(t(rv$values))
ans.tsne = Rtsne(t(rv$values[, !dups]), perplexity = 10)
cols = cols[!dups]
plot(ans.tsne$Y, col=cols)


d = svd(t(rv$values))$u[,1:2]
plot(d, col=cols)



mydata <- d
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:25) wss[i] <- sum(kmeans(mydata,
                                     centers=i)$withinss)
plot(1:25, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

require(cluster)
require(fpc)
pamk.best <- pamk(d)
cat("number of clusters estimated by optimum average silhouette width:", pamk.best$nc, "\n")
plot(pam(d, pamk.best$nc))



require(vegan)
fit <- cascadeKM(scale(d, center = TRUE,  scale = TRUE), 1, 30, iter = 1000)
plot(fit, sortg = TRUE, grpmts.plot = TRUE)
calinski.best <- as.numeric(which.max(fit$results[2,]))
cat("Calinski criterion optimal number of clusters:", calinski.best, "\n")




library(mclust)
# Run the function to see how many clusters
# it finds to be optimal, set it to search for
# at least 1 model and up 20.
d_clust <- Mclust(as.matrix(d), G=1:10)
m.best <- dim(d_clust$z)[2]
cat("model-based optimal number of clusters:", m.best, "\n")
# 4 clusters
plot(d_clust)



library(apcluster)
d.apclus <- apcluster(negDistMat(r=2), d)
cat("affinity propogation optimal number of clusters:", length(d.apclus@clusters), "\n")
# 4
heatmap(d.apclus)
plot(d.apclus, d)



library(NbClust)
nb <- NbClust(d, distance = "euclidean", 
              min.nc=2, max.nc=30, method = "kmeans", 
              index = "alllong", alphaBeale = 0.1)
hist(nb$Best.nc[1,], breaks = max(na.omit(nb$Best.nc[1,])))


