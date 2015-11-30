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


PIC_TOOL = convStr(Sys.getenv("PIC_TOOL"), "eog -f")

EP = convNum(Sys.getenv('EP'), -1)
WD = convStr(Sys.getenv('WD'), getwd())
T0 = convNum(Sys.getenv('T0'), 0)
T1 = convNum(Sys.getenv('T1'), 1000)

args <- commandArgs(trailingOnly = FALSE)
if(length(grep("RStudio", args))>0) {    
    WD = simruns.path(system(sprintf("ls -t %s | head -n 1", simruns.path()), intern=TRUE))
    WD = file.path(dnn.env(), "runs/test-run")
    #WD="/home/alexeyche/dnn/runs/cma_es/134"
    #Sys.setenv(CONST=file.path(WD, "dog_find.json"))
    
    system(sprintf("ls -t %s | head -n 1", WD))
    EP=as.numeric(strsplit(system(sprintf("basename $(ls -t %s/*.pb | head -n 1)", WD), intern=TRUE), "_")[[1]][1])
    #EP = 5
    #EP=2
}

pfx_f = function(s) s
if(EP>=0) {
    pfx_f = function(s) sprintf("%d_%s", EP, s)
}


CONST_FNAME = convStr(Sys.getenv('CONST'), "const.json")
MODEL_FNAME = convStr(Sys.getenv('MODEL'), pfx_f("model.pb"))
SPIKES_FNAME = convStr(Sys.getenv('SPIKES'), pfx_f("spikes.pb"))
INSP_SPIKES = convBool(Sys.getenv('INSP_SPIKES'), TRUE)
INSP_MODEL = convBool(Sys.getenv('INSP_MODEL'), TRUE)
EVAL_SPIKES_FNAME = convStr(Sys.getenv('EVAL_SPIKES'), pfx_f("eval_spikes.pb"))
STAT_FNAME = convStr(Sys.getenv('STAT'), pfx_f("stat.pb"))
SP_PIX0 = convNum(Sys.getenv('SP_PIX0'), 1024)
SP_PIX1 = convNum(Sys.getenv('SP_PIX1'), 768)
STAT_ID = convNum(Sys.getenv('STAT_ID'), 0) + 1 # C-like indices
STAT_SYN_ID = convNum(Sys.getenv('STAT_SYN_ID'), NULL)
COPY_PICS = convBool(Sys.getenv('COPY_PICS'), FALSE)
OPEN_PIC = convBool(Sys.getenv('OPEN_PIC'), TRUE)
LAYER_MAP = convStr(Sys.getenv('LAYER_MAP'), NULL)
SAVE_PIC_IN_FILES = convBool(Sys.getenv('SAVE_PIC_IN_FILES'), TRUE)
EVAL = convBool(Sys.getenv('EVAL'), TRUE)
EVAL_PROC = convStr(Sys.getenv('EVAL_PROC'), "Epsp(10)")
EVAL_KERN = convStr(Sys.getenv('EVAL_KERN'), "RbfDot(0.05)")
EVAL_JOBS = convNum(Sys.getenv('EVAL_JOBS'), 1)
EVAL_VERBOSE = convBool(Sys.getenv('EVAL_VERBOSE'), TRUE)
EVAL_TYPE = convStr(Sys.getenv('EVAL_TYPE'), "fisher")


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

if(file.exists(CONST_FNAME)) {
    const = fromJSON(readConst(CONST_FNAME))
    lsize = sapply(const$sim_configuration$layers, function(x) x$size)
    
    inputs = sapply(const$sim_configuration$files, function(x) x$filename)
    for(i in inputs) {
        i = gsub("^@", "", i)
        i = gsub("-", "_", i)
        ifile = sprintf("%s.pb", i)
        if(file.exists(ifile)) {
            if(!is.null(input)) {
                stop("Can't deal with multiple inputs")
            }
            input = RProto$new(ifile)$read()
        }
    }
}


pic_files = NULL

if(file.exists(EVAL_SPIKES_FNAME)) {
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
        model = RProto$new(MODEL_FNAME)$read()
        w = matrix(0, nrow=length(model), ncol=length(model))
        for(n in model) {
            w[n$id+1, n$synapses$ids_pre+1] = n$synapses$weights
        }
        
        weights_pic = sprintf("%s/2_%s", tmp_d, pfx_f("weights.png"))
        if(SAVE_PIC_IN_FILES) png(weights_pic, width=1024, height=768)
        print(gr_pl(w))
        if(SAVE_PIC_IN_FILES) { 
            dev.off()
            write(paste("Weights pic filename: ", weights_pic), stderr())
            pic_files = c(pic_files, weights_pic)
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

if (file.exists(STAT_FNAME)) {
    stat = RProto$new(STAT_FNAME)$rawRead()        
    stat_pic = sprintf("%s/3_%s", tmp_d, pfx_f("stat.png"))
    if(SAVE_PIC_IN_FILES) png(stat_pic, width=1024, height=768*6)
    
    if(length(stat)>=STAT_ID) {
        plot_stat(stat[[STAT_ID]], STAT_SYN_ID, T0, T1)
    } else {
        warning("STAT_ID is out of bounds")
    }
    if(SAVE_PIC_IN_FILES) {
        dev.off()
        write(paste("Stat pic filename: ", stat_pic), stderr())
        pic_files = c(pic_files, stat_pic)
    }
} else {
    warning(sprintf("Not found %s", STAT_FNAME))
}

cut_first_layer = function(sp) {
    first_layer_size = const$sim_configuration$layers[[1]]$size
    sp$values = sp$values[-(1:first_layer_size)] # w/o first layer
    return(sp)
}

if(EVAL) {
    source(scripts.path("eval.R"))
    
    if(!file.exists(SPIKES_FNAME)) {
        stop("Can't eval without spikes")        
    }
    
    setVerboseLevel(0)
    if(!is.null(spikes)) {
        eval_spikes = spikes     
    } else {
        eval_spikes = proto.read(SPIKES_FNAME)
    }
    
    if(sum(sapply(eval_spikes$values, length)) == 0) {
        cat("1.0\n")
    } else
    if(EVAL_TYPE == "fisher") {
        c(left_spikes, eval_spikes) := split.spikes(eval_spikes, length(eval_spikes$ts_info$labels_timeline)-floor(length(eval_spikes$ts_info$labels_timeline)/4))
        eval_spikes = cut_first_layer(eval_spikes)
        c(metric, K, y, M, N, A) := fisher_eval(eval_spikes, EVAL_VERBOSE)
        
        ans = K %*% y[, 1:2]
        eval_debug_pic = sprintf("%s/4_%s", tmp_d, pfx_f("eval.png"))
        if(SAVE_PIC_IN_FILES) png(eval_debug_pic, width=1024, height=768)
        
        par(mfrow=c(1,2))
        
        metrics_str = sprintf("%f", metric)
        plot(Re(ans[,1]), col=as.integer(rownames(K)), main=metrics_str) 
        plot(Re(ans), col=as.integer(rownames(K)))        
        
        if(SAVE_PIC_IN_FILES) {
            dev.off()
            write(paste("Eval debug pic filename: ", eval_debug_pic), stderr())
            pic_files = c(pic_files, eval_debug_pic)
        }
        par(mfrow=c(1,1))
        
        cat(metric, "\n") 
    } else
    if(EVAL_TYPE == "overlap") {
        c(left_spikes, eval_spikes) := split.spikes(eval_spikes, length(eval_spikes$ts_info$labels_timeline)-65)
        eval_spikes = cut_first_layer(eval_spikes)
        c(metric, vm) := overlap_eval(eval_spikes, const)
        
        cat(sprintf("%1.10f", metric), "\n")
        
        
        eval_ov_debug_pic = sprintf("%s/5_%s", tmp_d, pfx_f("eval_overlap.png"))
        
        if(SAVE_PIC_IN_FILES) png(eval_ov_debug_pic, width=1024, height=768)
        
        plot(vm[,1], type="l", main=sprintf("Metric: %f", metric), ylim=c(min(vm), max(vm)))
        for(li in 2:ncol(vm)) {
            lines(vm[,li], col=li)
        }
        if(SAVE_PIC_IN_FILES) {
            dev.off()
            write(paste("Eval overlap debug pic filename: ", eval_ov_debug_pic), stderr())
            pic_files = c(pic_files, eval_ov_debug_pic)
        }
    } else 
    if(EVAL_TYPE == "fisher_overlap") {
        c(metric_overlap, vm) := overlap_eval(eval_spikes, const)
        c(metric_fisher, K, y, M, N, A) := fisher_eval(eval_spikes, FALSE)
        cat(- abs(metric_overlap) * abs(metric_fisher), "\n")
    }
}

# if ( (!is.null(input))&&(!is.null(model))&&(!is.null(net)) ) {
#     PATTERN_LAYER = c(1)
#     
#     patterns = list()
#     last_pattern_time = 0
#     for(lt_i in 1:length(input$ts_info$labels_timeline)) {
#         lt = input$ts_info$labels_timeline[lt_i]
#         li = input$ts_info$labels_ids[lt_i]
#         lab = input$ts_info$unique_labels[li+1]
#         patterns[[lt_i]] = list()
#         patterns[[lt_i]]$values = blank_net(length(net))
#         patterns[[lt_i]]$label_id = li
#         for(ni in 1:length(net$values)) {
#             sp = net$values[[ni]]
#             for(sp_t in sp) {
#                 if((sp_t<last_pattern_time)||(sp_t>lt)) next
#                 patterns[[lt_i]]$values[[ni]] = c(patterns[[lt_i]]$values[[ni]], sp_t-last_pattern_time)
#             }        
#         }
#         last_pattern_time = lt
#     }
#     bin_patterns = list()
#     last_pattern_time = 0
#     for(lt_i in 1:length(input$ts_info$labels_timeline)) {
#         lt = input$ts_info$labels_timeline[lt_i]
#         li = input$ts_info$labels_ids[lt_i]
#         bin_patterns[[lt_i]] = list()
#         bin_patterns[[lt_i]]$label_id = li
#         dur = lt-last_pattern_time
#         bin_patterns[[lt_i]]$pattern = matrix(0, ncol=dur,nrow=length(patterns[[lt_i]]$values))
#         for(ni in 1:length(patterns[[lt_i]]$values)) {
#             bin_patterns[[lt_i]]$pattern[ni, patterns[[lt_i]]$values[[ni]] ] = 1
#         }
#         last_pattern_time = lt
#     }
#     lab_pattern = list()
#     map_patterns = list()
#     for(p in patterns) {
#         layers_maps = list()
#         for(ni in 1:length(model)) {
#             
#             n = model[[ni]]
#             if(n$localId == 0) layers_maps[[length(layers_maps)+1]] = matrix(0, nrow=n$colSize, ncol=n$colSize)
#             
#             layers_maps[[length(layers_maps)]][n$xi+1, n$yi+1] = length(p$values[[n$id+1]])
#             
#         }
#         map_patterns[[length(map_patterns)+1]] = list(map=layers_maps[[PATTERN_LAYER+1]], label=p$label)
#         if( (p$label_id+1) > length(lab_pattern) ) {
#             lab_pattern[[p$label_id+1]] = layers_maps[[PATTERN_LAYER+1]]
#         } else {
#             lab_pattern[[p$label_id+1]] = (lab_pattern[[p$label_id+1]] + layers_maps[[PATTERN_LAYER+1]])/2
#         }
#     }
#     lab_errors = matrix(0, nrow=length(map_patterns), ncol=length(lab_pattern))
#     for(pid in 1:length(map_patterns)) {
#         p = map_patterns[[pid]]
#         for(lpi in 1:length(lab_pattern)) {
#             lp = lab_pattern[[lpi]]
#             lab_errors[pi, lpi] = sum( (p$map - lp)^2 )
#         }
#     }
    
#     lab_errors_pic = sprintf("%s/5_%s", tmp_d, pfx_f("lab_errors.png"))
#     if(SAVE_PIC_IN_FILES) png(lab_errors_pic, width=1024, height=768)
#     print(gr_pl(lab_errors))
#     if(SAVE_PIC_IN_FILES) {
#         dev.off()
#         cat("Lab errors pic filename: ", lab_errors_pic, "\n")
#         pic_files = c(pic_files, lab_errors_pic)
#     }
#}


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

#plotl(get_stat(1:10, "OptimalStdp_w_0", 1))

annoying_file = file.path(getwd(), "Rplots.pdf")
if(file.exists(annoying_file)) {
    success = file.remove(annoying_file)
}
