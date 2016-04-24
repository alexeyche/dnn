

dnn.env = function () {
	dnn_env = Sys.getenv("DNN_HOME", "/usr/local/dnn")	
	if(!file.exists(dnn_env)) {
		stop("Can't find dnn home, make sure it is installed and environment varibale DNN_HOME points at a right place, or it is installed in system path /usr/local/dnn")
	}
	return(dnn_env)
}

ts.path = function(...) {
	file.path(dnn.env(), "ts", ...)
}

spikes.path = function(...) {
	file.path(dnn.env(), "spikes", ...)
}

ds.path = function(...) {
	file.path(dnn.env(), "datasets", ...)
}

scripts.path = function(...) {
    file.path(dnn.env(), "r_scripts", ...)
}

user.json.file = function() {
    file.path(dnn.env(), "scripts", "user.json")
}

runs.path = function(...) {
    file.path(dnn.env(), "runs", ...)
}

simruns.path = function(...) {
    runs.path("sim", ...)
}

read.state.script = function() {
    file.path(dnn.env(), "scripts", "read_state.py")
}

run.evolve.script = function() {
    file.path(dnn.env(), "scripts", "run_evolve.py")
}

read.spikes.wd = function(epoch=NULL) {
    if (is.null(epoch)) {
        epoch = as.numeric(strsplit(system("ls -t *.pb | head -n 1", intern=TRUE), "_")[[1]][1])    
    }
    
    eval_spikes_fname = sprintf("%d_eval_spikes.pb", epoch)
    spikes_fname = sprintf("%d_spikes.pb", epoch)
    
    spikes = NULL
    if (file.exists(eval_spikes_fname)) {
        spikes = proto.read(eval_spikes_fname)
    } else 
        if (file.exists(spikes_fname)) {
            spikes = proto.read(spikes_fname)    
        } else {
            stop(sprintf("Failed to find spikes in directory %d", getwd()))
        }
    return (list(spikes, epoch))    
}

read.input.ts.wd = function() {
    f = "input_time_series.pb"
    if (!file.exists(f)) {
        stop(sprintf("Can't find input time series file %s", f))
    }
    return(proto.read(f))
}
