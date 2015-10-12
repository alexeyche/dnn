

prepare.ucr.data = function(
    sample_size = 60
  , data_name = UCR.SYNTH
  , sel = NULL
  , gap_between_patterns = 0
  , dt = 1.0
  , save_on_disk=TRUE
  , use_cache=TRUE
) {

    c(data_train, data_test) := read_ucr_file(data_name, sample_size)
    data_complect = list(train=data_train, test=data_test)
    
    data_out = list()
    for(data_part in names(data_complect)) {
        ts = data_complect[[data_part]]
        sel.w = sel
        if(is.null(sel.w)) {
            sel.w = 1:length(ts)    
        }
        
        fname = sprintf(
            "%s/%s_%snum_%slen_%sclasses_%sgap_%s.pb"
          , ts.path()
          , data_name
          , length(sel.w)
          , sample_size
          , length(unique(sapply(ts, function(x) x$label)))
          , gap_between_patterns
          , data_part
        )
        if(use_cache && file.exists(fname)) {
            data_out[[data_part]] = proto.read(fname)
            cats("Using cached time series from %s\n", fname)
        } else {
            ts_info = list(labels_timeline=c())
            labels = c()
            
            time = 0
            ts_data = c()
            for(i in sample(sel.w)) {
                for(x in ts[[i]]$values) {
                    time = time + dt    
                }
                time = time + gap_between_patterns
                ts_info$labels_timeline = c(ts_info$labels_timeline, time)
                labels = c(labels, as.character(ts[[i]]$label))
                ts_data =c(ts_data, ts[[i]]$values, rep(0, gap_between_patterns))
            }
            ts_info$unique_labels = unique(labels)
            l_ids = list()
            idx = 0
            for(l in ts_info$unique_labels) {
                l_ids[[l]] = idx
                idx = idx + 1
            }
            ts_info$labels_ids = as.numeric(sapply(labels, function(l) l_ids[[l]]))
            data_out[[data_part]] = time.series(values=ts_data, ts_info=ts_info)
            if(save_on_disk) {
                cats("Saving %s in %s\n", data_part, fname)
                proto.write(data_out[[data_part]], fname)
            }
        }
    }
    return(data_out)
}


read_ucr_file <- function(ts_name, sample_size, ucr_dir=ds.path("ucr")) {
    process_datamatrix <- function(m) {
        l = ncol(m)
        out = list()
        for(ri in 1:nrow(m)) {
            out[[ri]] = list(values = m[ri,2:l], label = m[ri,1])    
        }  
        return(out)
    }
    if(!is.na(sample_size)) {
        ts_file_train = sprintf("%s/%s/%s_TRAIN_%s", ucr_dir, ts_name, ts_name, sample_size)
        ts_file_test = sprintf("%s/%s/%s_TEST_%s", ucr_dir, ts_name, ts_name, sample_size)
        if(!file.exists(ts_file_train)) {
            c(train_dataset, test_dataset) := read_ucr_file(ts_name, NA, ucr_dir)
            train_dataset_inter = matrix(0, length(train_dataset), sample_size+1)
            test_dataset_inter = matrix(0, length(test_dataset), sample_size+1)
            for(i in 1:length(train_dataset)) {
                inter_ts = interpolate_ts(train_dataset[[i]]$values, sample_size)
                train_dataset_inter[i, ] = c(train_dataset[[i]]$label,inter_ts)
            }
            for(i in 1:length(test_dataset)) {
                inter_ts = interpolate_ts(test_dataset[[i]]$values, sample_size)
                test_dataset_inter[i, ] = c(test_dataset[[i]]$label,inter_ts)
            }
            write.table(train_dataset_inter,file=ts_file_train,sep=" ", col.names = F, row.names = F, append=F)
            write.table(test_dataset_inter,file=ts_file_test,sep=" ", col.names = F, row.names = F)
        }
    } else {
        ts_file_train = sprintf("%s/%s/%s_TRAIN", ucr_dir, ts_name, ts_name)
        ts_file_test = sprintf("%s/%s/%s_TEST", ucr_dir, ts_name, ts_name)
    }
    nlines_train = as.numeric(system(sprintf("grep -c ^ %s", ts_file_train), intern=TRUE))
    nlines_test = as.numeric(system(sprintf("grep -c ^ %s", ts_file_test), intern=TRUE))
    ts_train = scan(ts_file_train)
    ts_test = scan(ts_file_test)
    
    ts_train = matrix(ts_train, nrow=nlines_train, byrow=TRUE)
    ts_test = matrix(ts_test, nrow=nlines_test, byrow=TRUE)
    return( list(process_datamatrix(ts_train), process_datamatrix(ts_test)) )
}

UCR.SYNTH = "synthetic_control"
UCR.ECG = "ECG200"
UCR.FACE = "FaceAll"
UCR.STARLIGH = "StarLightCurves"

empty.ts = function() {
    time.series(values=c(), ts_info=list(labels_ids=NULL, unique_labels=NULL, labels_timeline=NULL))
}

empty.spikes = function() {
    spikes.list(values=c(), ts_info=list(labels_ids=NULL, unique_labels=NULL, labels_timeline=NULL))
}

cat.ts = function(...) {
    ts = list(...)
    fts = empty.ts()
    fts$ts_info$unique_labels = unique(c(sapply(ts, function(x) x$ts_info$unique_labels)))
    
    last_t = 0
    for(t in ts) {
        fts$values = c(fts$values, t$values)
        labs = t$ts_info$unique_labels[ t$ts_info$labels_ids+1 ]
        
        fts$ts_info$labels_ids = c(fts$ts_info$labels_ids, sapply(labs, function(l) which(l == fts$ts_info$unique_labels)-1))
        
        fts$ts_info$labels_timeline = c(fts$ts_info$labels_timeline, t$ts_info$labels_timeline+last_t)     
        last_t = tail(t$ts_info$labels_timeline, 1)
    }
    return(fts)
}

split.spikes = function(sp, number_to_split) {
    time_to_split = sp$ts_info$labels_timeline[number_to_split] 
    left_idx = which(sp$ts_info$labels_timeline <= time_to_split)
    left_sp = empty.spikes()    
    right_sp = empty.spikes()
    
    left_sp$ts_info$labels_timeline = sp$ts_info$labels_timeline[left_idx]
    right_sp$ts_info$labels_timeline = sp$ts_info$labels_timeline[-left_idx] - time_to_split
    
    labs = sp$ts_info$unique_labels[ sp$ts_info$labels_ids+1 ]

    left_labs = labs[left_idx]
    left_sp$ts_info$unique_labels = unique(left_labs)
    left_sp$ts_info$labels_ids = sapply(left_labs, function(l) which(l == left_sp$ts_info$unique_labels)-1)
    left_sp$values = sapply(sp$values, function(spike_times) spike_times[ which(spike_times<=time_to_split) ] )
    
    right_labs = labs[-left_idx]
    right_sp$ts_info$unique_labels = unique(right_labs)
    right_sp$ts_info$labels_ids = sapply(right_labs, function(l) which(l == right_sp$ts_info$unique_labels)-1)
    right_sp$values = sapply(sp$values, function(spike_times) spike_times[ which(spike_times>time_to_split) ] - time_to_split)
    
    return(list(left_sp, right_sp))    
}

