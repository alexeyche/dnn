#!/usr/bin/env Rscript
library(utils, quietly=TRUE)
library(base, quietly=TRUE)
library(graphics, quietly=TRUE)
library(stats, quietly=TRUE)
library(datasets, quietly=TRUE)
library(grDevices, quietly=TRUE)
library(methods, quietly=TRUE)

require(Rdnn, quietly=TRUE)
require(rjson, quietly=TRUE)
#source(scripts.path("apply_user_env.R"))

PIC_TOOL = convStr(Sys.getenv("PIC_TOOL"), "eog -f")

EP = convNum(Sys.getenv('EP'), -1)
WD = convStr(Sys.getenv('WD'), getwd())
T0 = convNum(Sys.getenv('T0'), 0)
T1 = convNum(Sys.getenv('T1'), 1000)

args <- commandArgs(trailingOnly = FALSE)
if(length(grep("RStudio", args))>0) {    
    #WD = simruns.path(system(sprintf("ls -t %s | head -n 1", simruns.path()), intern=TRUE))
    WD = file.path(dnn.env(), "runs/last")
    
    #WD="/home/alexeyche/dnn/runs/evo_2l/66b0acb024f47e1a2de6c16c6bfc4df5_0342"
    
    system(sprintf("ls -t %s | head -n 1", WD))
    EP=as.numeric(strsplit(system(sprintf("basename $(ls -t %s/*.pb | head -n 1)", WD), intern=TRUE), "_")[[1]][1])
    #EP=2
}

pfx_f = function(s) s
if(EP>=0) {
    pfx_f = function(s) sprintf("%d_%s", EP, s)
}

CONFIG_FNAME = Sys.glob(file.path(WD, "*.pb.txt"))
MODEL_FNAME = convStr(Sys.getenv('MODEL'), pfx_f("model.pb"))
SPIKES_FNAME = convStr(Sys.getenv('SPIKES'), pfx_f("spikes.pb"))
INSP_SPIKES = convBool(Sys.getenv('INSP_SPIKES'), TRUE)
INSP_MODEL = convBool(Sys.getenv('INSP_MODEL'), TRUE)
EVAL_SPIKES_FNAME = convStr(Sys.getenv('EVAL_SPIKES'), pfx_f("eval_spikes.pb"))
STAT_FNAME = convStr(Sys.getenv('STAT'), pfx_f("stat.pb"))
EVAL_STAT_FNAME = convStr(Sys.getenv('EVAL_STAT'), pfx_f("eval_stat.pb"))
SP_PIX0 = convNum(Sys.getenv('SP_PIX0'), 1024)
SP_PIX1 = convNum(Sys.getenv('SP_PIX1'), 768)
STAT_ID = convNum(Sys.getenv('STAT_ID'), 0) + 1 # C-like indices
STAT_SYN_ID = convNum(Sys.getenv('STAT_SYN_ID'), NULL)
COPY_PICS = convBool(Sys.getenv('COPY_PICS'), FALSE)
OPEN_PIC = convBool(Sys.getenv('OPEN_PIC'), TRUE)
LAYER_MAP = convStr(Sys.getenv('LAYER_MAP'), NULL)
SAVE_PIC_IN_FILES = convBool(Sys.getenv('SAVE_PIC_IN_FILES'), TRUE)

if(length(grep("RStudio", args))>0) {
    STAT_ID=1
    STAT_SYN_ID=2
    LAYER_MAP= NULL #"1:0:0"
    SAVE_PIC_IN_FILES = FALSE    
}

setwd(WD)


tmp_d = Rdnn.tempdir()

input = NULL
model = NULL
net = NULL
spikes = NULL
pic_files = NULL

if(file.exists(EVAL_SPIKES_FNAME)) {
    eval_run_mode = TRUE
    SPIKES_FNAME = EVAL_SPIKES_FNAME
}
if(INSP_SPIKES) {
    if(file.exists(SPIKES_FNAME)) {
        spikes = proto.read(SPIKES_FNAME)
        net = spikes$values
        
        spikes_pic = sprintf("%s/1_%s", tmp_d, pfx_f("spikes.png"))
        if(SAVE_PIC_IN_FILES) png(spikes_pic, width=SP_PIX0, height=SP_PIX1)
        pspikes = plot(spikes, T0=T0,Tmax=T1)
        
        print(pspikes)
        
        if(SAVE_PIC_IN_FILES) {
            dev.off()
            
            write(paste("Spikes pic filename: ", spikes_pic), stderr())
            pic_files = c(pic_files, spikes_pic)
        }
    } else {
        warning(sprintf("Not found %s", SPIKES_FNAME))
    }
}

if(INSP_MODEL) {
    if (file.exists(MODEL_FNAME)) {
        model = proto.read(MODEL_FNAME)
        w = matrix(0, nrow=length(model), ncol=length(model))
        postW = matrix(0.0, nrow=length(model), ncol=length(model))
        
        post_w_makes_sense = FALSE
        for(n in model) {
            w[n$id+1, n$synapses$ids_pre+1] = n$synapses$weights
            postW[n$id+1, n$synapses$ids_pre+1] = n$synapses$post_synaptic_weights
            if ( (length(n$synapses$post_synaptic_weights)>0) && (!all(n$synapses$post_synaptic_weights == 1.0))) {
                post_w_makes_sense = TRUE                
            }
        }
        
        
        weights_pic = sprintf("%s/2_%s", tmp_d, pfx_f("weights.png"))
        if(SAVE_PIC_IN_FILES) png(weights_pic, width=1024, height=768)
        #lsize = 10
        #l1 = 10
        #l2 = (max(l1)+1):(max(l1)+lsize)
        #l3 = (max(l2)+1):(max(l2)+lsize)
        
        plw = function(ww, exc=TRUE) {
            if (exc) {
                ww[which(ww<0, arr.ind=TRUE)] <- 0
                print(levelplot(t(ww), col.regions = colorRampPalette(c("black", "white"))))
            } else {
                ww[which(ww>0, arr.ind=TRUE)] <- 0
                print(levelplot(t(ww), col.regions = colorRampPalette(c("white", "black"))))
            }
        }
        
        #print(gr_pl(t(w[l2, l1])))
        #plw(w[l2,l2], exc=TRUE)
        #plw(w[l2,l2], exc=FALSE)
        gr_pl(w)
        if(SAVE_PIC_IN_FILES) { 
            dev.off()
            write(paste("Weights pic filename: ", weights_pic), stderr())
            pic_files = c(pic_files, weights_pic)
        }
        
        if (post_w_makes_sense) {
            post_weights_pic = sprintf("%s/2_%s", tmp_d, pfx_f("post_weights.png"))
            if(SAVE_PIC_IN_FILES) png(weights_pic, width=1024, height=768)
            print(gr_pl(postW))
            if(SAVE_PIC_IN_FILES) { 
                dev.off()
                write(paste("Post weights pic filename: ", post_weights_pic), stderr())
                pic_files = c(pic_files, post_weights_pic)
            }
        }
        
        
        if(!is.null(LAYER_MAP)) {
            spl = as.numeric(strsplit(LAYER_MAP, ":")[[1]])        
            maps = getWeightMaps(spl[2]+1,spl[3]+1, w, lsize)
            weight_map_pic = sprintf("%s/4_%s", tmp_d, pfx_f("weight_map.png"))
            if(SAVE_PIC_IN_FILES) png(weight_map_pic, width=1024, height=768)
            print(gr_pl(maps[[spl[1]+1]]))
            if(SAVE_PIC_IN_FILES) {
                dev.off()
                pic_files = c(pic_files, weight_map_pic)
                cat("Weight map pic filename: ", weight_map_pic, "\n")
            }
        }
    } else {
        warning(sprintf("Not found %s", MODEL_FNAME))
    }   
}

if(file.exists(EVAL_STAT_FNAME)) {
    eval_run_mode = TRUE
    STAT_FNAME = EVAL_STAT_FNAME
}
if (file.exists(STAT_FNAME)) {
    stat = proto.read(STAT_FNAME)
    stat_pic = sprintf("%s/3_%s", tmp_d, pfx_f("stat.png"))
    if(SAVE_PIC_IN_FILES) png(stat_pic, width=1024, height=768*6)
    stat_to_plot = stat
    if (length(stat_to_plot) > 8) {
        stat_to_plot = stat[1:8]
    }    
    par(mfrow=c(length(stat_to_plot),1), mar=rep(2,4))
    for (s in stat_to_plot) {
        plot(
            seq(s$from, s$to, length.out=length(s$values)), 
            s$values, 
            type="l", 
            main=sprintf("%s", s$name),
            xlab="Time", ylab=s$name
        )
    }
    if(SAVE_PIC_IN_FILES) {
        dev.off()
        write(paste("Stat pic filename: ", stat_pic), stderr())
        pic_files = c(pic_files, stat_pic)
    }
    par(mfrow=c(1,1))
} else {
    warning(sprintf("Not found %s", STAT_FNAME))
}

cut_first_layer = function(sp) {
    first_layer_size = const$sim_configuration$layers[[1]]$size
    sp$values = sp$values[-(1:first_layer_size)] # w/o first layer
    return(sp)
}



if(COPY_PICS) {
    new_pic_files = NULL
    for(p in pic_files) {
        dst = sprintf("%s/%s", getwd(), basename(p))
        file.copy(p, dst)
        new_pic_files = c(new_pic_files, dst)
    }
    pic_files = new_pic_files
}

if((length(pic_files)>0)&&(OPEN_PIC)) {
    open_pic(pic_files[1])
}

get_stat = function(epochs, stname, stat_id, f_template = "%s_stat.pb") {
    stat_acc = NULL
    for(ep in epochs) { 
        s = RProto$new(sprintf(f_template, ep))$rawRead()
        stat_acc = c(stat_acc, s[[stat_id]][[stname]])
    }
    return(stat_acc)
}

m.sort = function(arr) arr[do.call(order, lapply(1:ncol(arr), function(i) arr[, i])), ]

#plotl(get_stat(1:10, "OptimalStdp_w_0", 1))

annoying_file = file.path(getwd(), "Rplots.pdf")
if(file.exists(annoying_file)) {
    success = file.remove(annoying_file)
}
# if (exists("signal")) {
#     par(mfrow=c(1,1))
#     plot(neuron$weights, type="l", ylim=c(-0.2, 1.0))
#     ei = eigen(t(signal) %*% (signal))
#     lines(Re(ei$vectors[,1]), col="blue")
#     lines(w[nrow(w), which(sapply(spikes$values, length) > 0)], col="red",type="l")
#     gr_pl(t(m.sort(w[257:nrow(w),1:256])))
#     gr_pl(t(m.sort(t(abs(ica.signal$M)))))
# }

sigmoid = function(x, tt=0.1, s=100) {
    1/(1+exp(-(x-tt)/s))
}
logexp = function(x, t=0.1, s=1.0) {
    log( (1 + exp((x-t)/s))/(1 + exp((-t)/s)))
}

# for (ep in 1:100) {
#     model = proto.read(sprintf("%d_model.pb",ep))
#     w = matrix(0, nrow=length(model), ncol=length(model))
#     postW = matrix(0.0, nrow=length(model), ncol=length(model))
#     
#     for(n in model) {
#         w[n$id+1, n$synapses$ids_pre+1] = n$synapses$weights
#     }
#     print(gr_pl(w[257:266,257:266]))
# }
