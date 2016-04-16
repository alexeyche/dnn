#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

#include <dnn/util/log/log.h>
#include <dnn/util/ts/time_series.h>
#include <dnn/util/ts/spikes_list.h>

#include "proto.h"

// [[Rcpp::export(name = "set.verbose.level")]]
void setVerboseLevel(int level) {
    if(level == 0) {
        TLog::Instance().SetLogLevel(TLog::INFO_LEVEL);
    } else
    if(level == 1) {
        TLog::Instance().SetLogLevel(TLog::DEBUG_LEVEL);
    } else {
        ERR("Available verbose levels : 0 -- INFO, 1 -- DEBUG");
    }
}


// [[Rcpp::export(name = "chop.time.series")]]
Rcpp::List chopTimeSeries(Rcpp::List l) {
	TTimeSeries ts = TProto::TranslateBack<TTimeSeries>(l);

	Rcpp::List out;
	for(auto &ts_ch: ts.Chop()) {
		out.push_back(TProto::Translate<TTimeSeries>(ts_ch));
	}
	return out;
}

// [[Rcpp::export(name = "chop.spikes.list")]]
Rcpp::List chopSpikesList(Rcpp::List l) {
    TSpikesList ts = TProto::TranslateBack<TSpikesList>(l);

	Rcpp::List out;
	for(auto &ts_ch: ts.Chop()) {
		out.push_back(TProto::Translate<TSpikesList>(ts_ch));
	}
	return out;
}

// [[Rcpp::export(name = "binarize.spikes")]]
Rcpp::List binarizeSpikes(Rcpp::List l, double dt = 1.0) {
	TSpikesList ts = TProto::TranslateBack<TSpikesList>(l);
	return TProto::Translate<TTimeSeries>(ts.ConvertToBinaryTimeSeries(dt));
}