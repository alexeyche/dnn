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
    
    #WD="/home/alexeyche/dnn/runs/param_range/c230c7d64d9094a83c1ec9fc6a656def_0019"
    
    system(sprintf("ls -t %s | head -n 1", WD))
    EP=as.numeric(strsplit(system(sprintf("basename $(ls -t %s/*.pb | head -n 1)", WD), intern=TRUE), "_")[[1]][1])
    #EP=4
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
EVAL = convBool(Sys.getenv('EVAL'), FALSE)
EVAL_PROC = convStr(Sys.getenv('EVAL_PROC'), "Epsp(10)")
EVAL_KERN = convStr(Sys.getenv('EVAL_KERN'), "RbfDot(0.05)")
EVAL_JOBS = convNum(Sys.getenv('EVAL_JOBS'), 1)
EVAL_VERBOSE = convBool(Sys.getenv('EVAL_VERBOSE'), TRUE)
EVAL_TYPE = convStr(Sys.getenv('EVAL_TYPE'), "linear_classifier")


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
eval_run_mode = FALSE

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
        #print(gr_pl(t(w[257:(256+10),1:256])))
        #print(gr_pl(t(w[257:nrow(w),257:nrow(w)])))
        
        #print(gr_pl(w))
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
            main=sprintf("%s, %d:%d", s$name, s$from, s$to),
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

if(EVAL) {
    source(scripts.path("eval.R"))
    
    if(!file.exists(SPIKES_FNAME)) {
        stop("Can't eval without spikes")        
    }
    
    set.verbose.level(1)
    if(!is.null(spikes)) {
        eval_spikes = spikes     
    } else {
        eval_spikes = proto.read(SPIKES_FNAME)
    }
    
    if(sum(sapply(eval_spikes$values, length)) == 0) {
        cat("1.0\n")
    } else
    if(EVAL_TYPE == "fisher") {
        #if(!eval_run_mode) {
        #    c(left_spikes, eval_spikes) := split.spikes(eval_spikes, length(eval_spikes$info)-floor(length(eval_spikes$info)/4))
        #}
        #eval_spikes = cut_first_layer(eval_spikes)
        eval_spikes$values = eval_spikes$values[-(1:100)]
        c(metric, K, y, M, N, A) := fisher_eval(eval_spikes, EVAL_VERBOSE, EVAL_JOBS)
        
        ans = K[1:50, 1:50] %*% y[1:50, 1:2]
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
        if(!eval_run_mode) {
            c(left_spikes, eval_spikes) := split.spikes(eval_spikes, length(eval_spikes$info)-floor(length(eval_spikes$info)/4))
        }
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
    } else
      if (EVAL_TYPE == "error_rate") {
        # error rate, density of probability and ROC-curve (required package "ROCR")
        print.roc = FALSE
        if ("ROCR" %in% installed.packages()[,"Package"]) {
          require("ROCR")
          print.roc = TRUE
        }
        chop.spikes = chop.spikes.list(spikes)
        errors = c()
        classes = c()
        errors.count = 0
        probability.list = list()
        
        first.neuron = length(spikes$values) - const$sim_configuration$layers[[length(const$sim_configuration$layers)]]$size
        last.neuron = length(spikes$values)
        
        labels.vec = c()
        for (i in 1: length(spikes$info)) {
          labels.vec = c(labels.vec, spikes$info[[i]]$label)
        }
        labels.vec = unique(labels.vec)
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
        eval_debug_pic = sprintf("%s/4_%s", tmp_d, pfx_f("density.png"))
        if(SAVE_PIC_IN_FILES) png(eval_debug_pic, width = 1024, height = 768)
        if (print.roc) {
          par(mfrow = c((last.neuron - first.neuron), (last.neuron - first.neuron)))
        } else {
          par(mfrow = c((last.neuron - first.neuron - 1), (last.neuron - first.neuron - 1)))
        }
        # errors rate in current epoch
        error.rate = errors[[length(errors)]]/length(errors)
        plot(errors, main = sprintf("Epoch #%d, error rate %.3f", EP, error.rate),
             ylab = "Number of errors", type = "s", col = "darkred", lwd = 2)

        for (n in 1:(last.neuron - first.neuron)) {
          if (n != (last.neuron - first.neuron)) {
            for (m in (n + 1):(last.neuron - first.neuron)) {
              plot(1, type = "n", ylab = "Density", xlab = "Probability", xlim = 0:1, 
                   ylim = c(0,max(c(max(density(probability.list[[m]][, m])$y), max(density(probability.list[[n]][, n])$y)))),
                   main = sprintf("Densities, classes %s and %s", labels.vec[n], labels.vec[m]))
              lines(density(probability.list[[n]][, n]), xlim = 0:1, lty = 1, col = n, lwd = 2)
              lines(density(1 - probability.list[[m]][, m]), xlim = 0:1, lty = 1, col = m, lwd = 1.5)
              legend("topleft", bty = "n", c(labels.vec[n], labels.vec[m]),
                     lty = c(1, 1), lwd = c(2, 1.5), col = c(n, m))
              grid(nx = 2, ny = NA, lty = 1, lwd = 2)
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
              }
            }
          }
        }

        cat(sprintf("%1.10f", error.rate), "\n")
        if(SAVE_PIC_IN_FILES) {
          dev.off()
          write(paste("Eval debug pic filename: ", eval_debug_pic), stderr())
          pic_files = c(pic_files, eval_debug_pic)
        }
      }
    if(EVAL_TYPE == "linear_classifier") {
        input_neurons = 100
        rates = NULL
        labs = NULL
        for (sp in chop.spikes.list(spikes)) {
            rates = rbind(rates, sapply(sp$values[-(1:input_neurons)], length)/sp$info[[1]]$duration)
            labs = c(labs, sp$info[[1]]$label)
        }
        K = rates %*% t(rates)
        colnames(K) <- labs
        c(y, M, N, A) := KFD(K)
        metric = -tr(M)/tr(N)
        
        ans = K %*% y[, 1:2]
        eval_debug_pic = sprintf("%s/4_%s", tmp_d, pfx_f("eval.png"))
        if(SAVE_PIC_IN_FILES) png(eval_debug_pic, width=1024, height=768)
        
        par(mfrow=c(1,2))
        
        metrics_str = sprintf("%f", metric)
        plot(Re(ans[,1]), col=as.integer(colnames(K)), main=metrics_str) 
        plot(Re(ans), col=as.integer(colnames(K)))        
        
        if(SAVE_PIC_IN_FILES) {
            dev.off()
            write(paste("Eval debug pic filename: ", eval_debug_pic), stderr())
            pic_files = c(pic_files, eval_debug_pic)
        }
        par(mfrow=c(1,1))
        
        cat(metric, "\n") 
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



