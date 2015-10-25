
.onLoad <- function(pkgname, libname) {
    assign("RSim", Module("dnnMod")$RSim, envir=parent.env(environment()))
    assign("RMatchingPursuit", Module("dnnMod")$RMatchingPursuit, envir=parent.env(environment()))
    assign("RProto", Module("dnnMod")$RProto, envir=parent.env(environment()))
    assign("RConstants", Module("dnnMod")$RConstants, envir=parent.env(environment()))
    assign("RGammatoneFB", Module("dnnMod")$RGammatoneFB, envir=parent.env(environment()))
    assign("RKernel", Module("dnnMod")$RKernel, envir=parent.env(environment()))
    setMethod( "show", RSim, function(object) {
        object$print()
    } )
    setMethod( "show", RConstants, function(object) {
        object$print()
    } )
    setMethod( "show", RProto, function(object) {
        object$print()
    } )
    setMethod( "show", RGammatoneFB, function(object) {
        object$print()
    } )
    setMethod( "show", RMatchingPursuit, function(object) {
        object$print()
    } )
    setMethod( "show", RKernel, function(object) {
        object$print()
    } )
}


proto.read = function(f) {
    f = path.expand(f)
    return(Module("dnnMod")$RProto$new(f)$read())
}


proto.write = function(l, f) {
    f = path.expand(f)
    return(Module("dnnMod")$RProto$new(f)$write(l))
}



get.gammatones = function(freqs, samp_rate, len=100) {
    inp = c(1, rep(0, len-1))
    gfb = Module("dnnMod")$RGammatoneFB$new()
    gfb_out = gfb$calc(inp, freqs, samp_rate, 0, 0)
    return(gfb_out$membrane)
}

time.series = function(values, ts_info=NULL) {
    o = list(values = values)
    if(is.null(ts_info)) {
        if(is.matrix(values)) {
            l = ncol(values)    
        } else {
            l = length(values)
        }
        ts_info = ts.info("unknown_label", l)
    }
    o$ts_info = ts_info
    class(o) <- "TimeSeries"
    return(o)
}

spikes.list = function(values, ts_info=list()) {
    o = list(values = values, ts_info = ts_info)
    class(o) <- "SpikesList"
    return(o)
}

conv.gammatones = function(x, freqs, samp_rate) {
    data = x
    info = NULL
    if("values" %in% names(x)) {
        data = x$values
        info = x$ts_info
    }

    gfb = Module("dnnMod")$RGammatoneFB$new()
    gfb_out = gfb$calc(data, freqs, samp_rate, 0, 0)
    if(!is.null(info)) {
        return(time.series(values=gfb_out$membrane, ts_info = info))
    }
    return(time.series(gfb_out$membrane))
}

kernel.run = function(data=NULL, preprocessor=NULL, kernel=NULL, jobs=1) {
    if((is.null(kernel) && is.null(preprocessor)) || is.null(data)) {
        cat("Usage of kernel.run:\n\tkernel.run([list with data], [preprocessor string spec], [kernel string spec]). Need preprocessor or kernel spec or both\n")
        RKernel$new()$print()
        stop("Need data, kernel or preprocessor not empty. Read specs and point some stuff in there")
    }
    specs = list()
    if(!is.null(kernel)) {
        specs$kernel = kernel
    }
    if(!is.null(preprocessor)) {
        specs$preprocessor = preprocessor
    }
    specs$jobs = jobs
    RKernel$new()$run(data, specs)
}
