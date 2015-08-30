
.onLoad <- function(pkgname, libname) {
    assign("RSim", Module("dnnMod")$RSim, envir=parent.env(environment()))
    assign("RMatchingPursuit", Module("dnnMod")$RMatchingPursuit, envir=parent.env(environment()))
    assign("RProto", Module("dnnMod")$RProto, envir=parent.env(environment()))
    assign("RConstants", Module("dnnMod")$RConstants, envir=parent.env(environment()))
    assign("RGammatoneFB", Module("dnnMod")$RGammatoneFB, envir=parent.env(environment()))
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
    
}

proto.read = function(f) {
    f = path.expand(f)
    return(Module("dnnMod")$RProto$new(f)$read())
}

proto.write = function(f, l, n) {
    f = path.expand(f)
    return(Module("dnnMod")$RProto$new(f)$write(l, n))
}    

get.gammatones = function(freqs, samp_rate, len=100) {
    inp = c(1, rep(0, len-1))
    gfb = Module("dnnMod")$RGammatoneFB$new()
    gfb_out = gfb$calc(inp, freqs, samp_rate, 0, 0)
    return(gfb_out$membrane)
}


conv.gammatones = function(x, freqs, samp_rate) {
    gfb = Module("dnnMod")$RGammatoneFB$new()
    gfb_out = gfb$calc(x, freqs, samp_rate, 0, 0)
    return(gfb_out$membrane)
}


