
require(Rdnn)

ornstein_uhlenbeck <- function(T, n, nu,lambda,sigma,x0){
  dt  <- T/n
  dw  <- rnorm(n, 0, sqrt(dt))
  x <- c(x0)
  for (i in 2:n) {
    x[i]  <-  x[i-1] + lambda*(nu-x[i-1])*dt + sigma*dw[i-1]
  }
  return(x);
}

#set.seed(2)

sample_len = 500
pause = 100
v = NULL
info = list()
lab = "a"
for(i in 1:100) {
    v = c(v, ornstein_uhlenbeck(100, sample_len, 1.0, 0.01, 0.1, 0.0))
    v = c(v, rep(-100.0, pause))
    info = add.to.list(info, ts.info(duration=sample_len, label=lab, start_time=length(v)-sample_len-pause))
}


normalize = function(x, min_val, max_val) {
    2*( 1 - (max_val-x)/(max_val-min_val) ) -1
}


#v = normalize(v, min(v), max(v))
plot(v[1:sample_len], type="l", col="blue")


require(Rdnn)
proto.write(time.series(v, info), ts.path("test.pb"))