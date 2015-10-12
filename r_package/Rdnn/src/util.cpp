
#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

#include <dnn/util/log/log.h>
#include <dnn/base/factory.h>
#include <dnn/util/time_series.h>

#include "common.h"
#include "util.h"
#include "RProto.h"

using namespace dnn;



// [[Rcpp::export]]
void setVerboseLevel(int level) {
    if(level == 0) {
        Log::inst().setLogLevel(Log::INFO_LEVEL);
    } else
    if(level == 1) {
        Log::inst().setLogLevel(Log::DEBUG_LEVEL);
    } else {
        ERR("Available verbose levels : 0 -- INFO, 1 -- DEBUG");
    }
}


// [[Rcpp::export]]
void dnnCleanHeap() {
    Factory::inst().cleanHeap();
}


// [[Rcpp::export]]
Rcpp::List chopTimeSeries(Rcpp::List l) {
	Ptr<TimeSeries> ts = RProto::convertBack<TimeSeries, DynamicCreationPolicy>(l, "TimeSeries");
	vector<Ptr<TimeSeries>> chopped = ts->chop<DynamicCreationPolicy>();
	Rcpp::List out;

	for(auto &ts_ch: chopped) {
		SEXP ts_ch_r = RProto::convertToR(ts_ch.as<SerializableBase>());
		out.push_back(ts_ch_r);
		ts_ch.destroy();
	}
	ts.destroy();

	return out;
}