#ifndef RSPIKEWORK_READ_H
#define RSPIKEWORK_READ_H

#include <dnn/base/base.h>
#include <dnn/util/ts/time_series.h>
#include <dnn/util/ts/spikes_list.h>
#include <dnn/spikework/spikework.h>
#include <dnn/spikework/protos/spikework_config.pb.h>

#include <fstream>

#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

#include "common.h"
#include "proto.h"

using namespace NDnn;


// [[Rcpp::export(name = "pp.kernel.run")]]
Rcpp::NumericMatrix PpKernelRun(
	Rcpp::List preProcConfig, 
	Rcpp::List kernelConfig, 
	Rcpp::List timeSeries, 
	size_t jobs
) {
	TTimeSeries ts;
	TString type = timeSeries.attr("class");
	if (type == "SpikesList") {
		ts = TProto::TranslateBack<TSpikesList>(timeSeries).ConvertToBinaryTimeSeries(1.0);
	} else {
		ts = TProto::TranslateBack<TTimeSeries>(timeSeries); 
	}
	return TProto::Translate<TDoubleMatrix>(
		TSpikework::KernelRun(
			TProto::TranslateBack<NDnnProto::TPreprocessorConfig>(preProcConfig), 
			TProto::TranslateBack<NDnnProto::TKernelConfig>(kernelConfig), 
			ts, 
			jobs
		)
	);
}

// [[Rcpp::export(name = "kernel.run")]]
Rcpp::NumericMatrix KernelRun(
	Rcpp::List kernelConfig, 
	Rcpp::List timeSeries, 
	size_t jobs
) {
	return TProto::Translate<TDoubleMatrix>(
		TSpikework::KernelRun(
			TProto::TranslateBack<NDnnProto::TKernelConfig>(kernelConfig), 
			TProto::TranslateBack<TTimeSeries>(timeSeries), 
			jobs
		)
	);
}


// [[Rcpp::export(name = "preprocess.run")]]
Rcpp::List PreprocessRun(
	Rcpp::List preProcConfig,
	Rcpp::List timeSeries,
	size_t jobs
) {
	return TProto::Translate<TTimeSeries>(
		TSpikework::PreprocessRun(
			TProto::TranslateBack<NDnnProto::TPreprocessorConfig>(preProcConfig), 
			TProto::TranslateBack<TTimeSeries>(timeSeries), 
			jobs
		)
	);
}


#endif