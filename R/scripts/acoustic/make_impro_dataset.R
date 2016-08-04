
setwd(path.expand("~/Music/ml/jp"))


data = spikes.list(vector("list", 256))

lab = 0
for (f in system("ls *.pb", intern=TRUE)) {
    if (length(grep("_raw", f)) == 0) {
        cat(f, "\n")    
        sp = proto.read(f)
        sp = add.ts.info(sp, 
            ts.info(
                label=as.character(lab), 
                start_time=0, 
                duration=max(sapply(sp$values, function(x) if (length(x)>0) {max(x)} else{0}))
            )
        )
        data = add.to.spikes(data, sp)
        lab = lab + 1
    }
}

work_data = data

while (TRUE) {
    max_t = spikes.list.max.t(work_data)
    if (max_t > 200000) {
        break
    }
    work_data = add.to.spikes(work_data, work_data)
}


proto.write(data, spikes.path("impro_eval.pb"))
proto.write(work_data, spikes.path("impro.pb"))
