

require(Rdnn)

#cand = RProto$new("/home/alexeyche/dnn/build/left.pb")$read()$values[[1]]
#L = length(cand)
#vr1 = RProto$new("/home/alexeyche/dnn/build/right0_68.pb")$read()$values[[1]]

ts = RProto$new("/home/alexeyche/dnn/ts/synthetic_control_norm_40_len_4_classes_train.pb")$read()$values[[1]]
#ts = RProto$new("/home/alexeyche/dnn/ts/synthetic_control_norm_6_len_2_classes_train.pb")$read()$values[[1]]


cand = RProto$new("/home/alexeyche/dnn/build/best.pb")$read()$values[[1]]
L = length(cand)


dd = c()
dd2 = c()
for(i in 1:(length(ts)-L)) {
    r = ts[i:(i+L-1)]
    
    r = (r - mean(r))/sd(r)
    cand = (cand - mean(cand))/sd(cand)

    dd2 = c(dd2, sqrt(mean((r - cand)^2)))
    
    mean_prod = mean(cand * r)
    ml = mean(cand)
    mr = mean(r)
    ml2 = mean(cand^2)
    mr2 = mean(r^2)
    cov = mean_prod - ml*mr
    sd2l = ml2 - ml*ml
    sd2r = mr2 - mr*mr
    corr = cov/sqrt(sd2l*sd2r)
    dist2 = 2.0*(1.0 - corr)
    if(dist2<0) {
        dist2 = 0
    }
    dd = c(dd, sqrt(dist2))
}
