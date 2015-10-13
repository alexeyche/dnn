
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


// [[Rcpp::export(name = "dnn.clean.heap")]]
void dnnCleanHeap() {
    Factory::inst().cleanHeap();
}


// [[Rcpp::export(name = "chop.time.series")]]
Rcpp::List chopTimeSeries(Rcpp::List l) {
	Ptr<TimeSeries> ts = RProto::convertFromR<TimeSeries, DynamicCreationPolicy>(l);
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

// [[Rcpp::export(name = "chop.spikes.list")]]
Rcpp::List chopSpikesList(Rcpp::List l) {
    Ptr<SpikesList> sl = RProto::convertFromR<SpikesList, DynamicCreationPolicy>(l);
    vector<Ptr<SpikesList>> chopped = sl->chop<DynamicCreationPolicy>();
    Rcpp::List out;

    for(auto &sl_ch: chopped) {
        SEXP sl_ch_r = RProto::convertToR(sl_ch.as<SerializableBase>());
        out.push_back(sl_ch_r);

        sl_ch.destroy();
    }
    sl.destroy();

    return out;
}