
require(Rdnn)
require(DiceOptim)
require(DiceEval)
require(fields)
require(sensitivity)

run_name = "mc"

set.seed(13)

cross_validate = FALSE
#cross_validate = TRUE

o = system(sprintf("%s %s", read.state.script(), runs.path(run_name)), intern=TRUE)
M = t(sapply(strsplit(o, ","), as.numeric))

X = M[, 1:(ncol(M)-1)]
Y = M[, ncol(M)]
Y = -log(-Y)

X = X[,1:2]


#sel = sample(1:nrow(X), 50)
#X = X[sel, ]
#Y = Y[sel]

colnames(X) <- sprintf("x%d", 1:ncol(X))
if(!cross_validate) {
    m = km(
        ~ .
        , design = as.data.frame(X)
        , response = data.frame(Y=Y)
        , optim.method = "gen"
        #, covtype="gauss", nugget = 1e-8 * var(Y)
        #, covtype = "powexp"
        , control = list(pop.size = 30, max.generations = 30, wait.generations = 5, BFGSburnin = 2)
    )    
} else {
    mKm = modelFit(
        X, Y, type="Kriging"
        , formula = ~ .
        , optim.method="gen"
    )
    K=30
    out = crossValidation(mKm, K)
    
    m = mKm$model
    
    par(mfrow=c(2,2))
    plot(c(0,1:K),c(mKm$model@covariance@range.val[1],out$theta[,1]),xlab='',ylab='Theta1')
    plot(c(0,1:K),c(mKm$model@covariance@range.val[2],out$theta[,2]),xlab='',ylab='Theta2')
    #plot(c(0,1:K),c(mKm$model@covariance@range.val[1],out$shape[,1]),xlab='',ylab='p1')
    #plot(c(0,1:K),c(mKm$model@covariance@range.val[2],out$shape[,2]),xlab='',ylab='p2')
}



# par(mfrow = c(1, 4))
# llh.x.grid <- seq(0.01, 2, length = n.grid)
# llh.X.grid <- expand.grid(llh.x.grid, llh.x.grid)
# logLik.grid <- apply(llh.X.grid, 1, logLikFun, m)
# contour(llh.x.grid, llh.x.grid, matrix(logLik.grid, n.grid, n.grid), 40, xlab = expression(theta[1]), ylab = expression(theta[2]), main="Kriging LLH")
# opt <- m@covariance@range.val
# points(opt[1], opt[2], pch = 19, col = "red")


n.grid <- 10

par(mfrow = c(1, 3))

n.grid.sm <- 10

x.grid <- seq(0, 1, length = n.grid)
x.grid <- seq(0, 1, length = n.grid)
x.grid.sm <- seq(0, 1, length = n.grid.sm)

important_axes = c(1,3)

grid_list = lapply(1:ncol(X), function(i) if(i %in% important_axes) { x.grid } else { x.grid.sm })
names(grid_list) <- colnames(X)

X.grid <- expand.grid(grid_list)
pred.m <- predict(m, X.grid, "UK")
pred.m.mean = matrix(pred.m$mean, n.grid, n.grid)
pred.m.sd = matrix(pred.m$sd^2, n.grid, n.grid)

image.plot(x.grid, x.grid, pred.m.mean, main = "Kriging mean")
points(X[ , important_axes[1]], X[ , important_axes[2]], pch = 19, cex = 1.0, col = "white")
image.plot(x.grid, x.grid, pred.m.sd, main = "Kriging variance")
points(X[ , important_axes[1]], X[ , important_axes[2]], pch = 19, cex = 1.0, col = "white")

kriging.mean <- function(Xnew)  {
    predict.km(m, Xnew, "UK", se.compute = FALSE, checkNames = FALSE)$mean
}

SA.metamodel <- fast99(
    model = kriging.mean
  , factors = ncol(X)
  , n = 1000
  , q = "qunif"
  , q.arg = list(min = 0, max = 1)
)

plot(SA.metamodel)
