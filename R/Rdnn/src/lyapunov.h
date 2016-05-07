#ifndef RSPIKEWORK_READ_H
#define RSPIKEWORK_READ_H

#include <dnn/base/base.h>
#include <dnn/util/ts/time_series.h>
#include <dnn/util/ts/spikes_list.h>
#include <dnn/spikework/spikework.h>
#include <dnn/spikework/protos/spikework_config.pb.h>
#include <dnn/metrics/lyapunov.h>

#include <fstream>

#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

#include "common.h"
#include "proto.h"

using namespace NDnn;


// [[Rcpp::export(name = "lyapunov")]]
Rcpp::List Lyapunov( 
	Rcpp::List timeSeries,
	Rcpp::List outputSpikes
) {
	TTimeSeries ts_input;
	TString type_input = timeSeries.attr("class");
	if (type_input == "SpikesList") {
		ts_input = TProto::TranslateBack<TSpikesList>(timeSeries).ConvertToBinaryTimeSeries(1.0);
	} else {
		ts_input = TProto::TranslateBack<TTimeSeries>(timeSeries); 
	}
	TTimeSeries ts_output;
	TString type_output = outputSpikes.attr("class");
	if (type_output == "SpikesList") {
		ts_output = TProto::TranslateBack<TSpikesList>(outputSpikes).ConvertToBinaryTimeSeries(1.0);
	} else {
		ts_output = TProto::TranslateBack<TTimeSeries>(outputSpikes); 
	}
	return TProto::Translate<TTimeSeries>(
		TLyapunov::CalculateMetrics(
			ts_input, 
			ts_output 
		)
	);
}

#endif