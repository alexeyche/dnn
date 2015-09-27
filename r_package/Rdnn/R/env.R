

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