#ifndef RSPIKEWORK_READ_H
#define RSPIKEWORK_READ_H

#include <ground/base/base.h>
#include <ground/ts/time_series.h>
#include <ground/ts/spikes_list.h>
#include <dnn/spikework/spikework.h>
#include <dnn/spikework/protos/spikework_config.pb.h>

#include <fstream>

#undef PI
#define STRICT_R_HEADERS
#include <Rcpp.h>

#include "common.h"
#include "proto.h"

using namespace NDnn;
using namespace NGround;


// [[Rcpp::export(name = "pp.class.kernel.run")]]
Rcpp::NumericMatrix PpClassKernelRun(
	Rcpp::List preProcConfig, 
	Rcpp::List kernelConfig, 
	Rcpp::List timeSeries, 
	size_t jobs,
	double dt = 1.0
) {
	TTimeSeries ts;
	TString type = timeSeries.attr("class");
	if (type == "SpikesList") {
		ts = TProto::TranslateBack<TSpikesList>(timeSeries).ConvertToBinaryTimeSeries(dt);
	} else {
		ts = TProto::TranslateBack<TTimeSeries>(timeSeries); 
	}
	return TProto::Translate<TDoubleMatrix>(
		TSpikework::ClassKernelRun(
			TProto::TranslateBack<NDnnProto::TPreprocessorConfig>(preProcConfig), 
			TProto::TranslateBack<NDnnProto::TKernelConfig>(kernelConfig), 
			ts, 
			jobs
		)
	);
}

// [[Rcpp::export(name = "class.kernel.run")]]
Rcpp::NumericMatrix ClassKernelRun(
	Rcpp::List kernelConfig, 
	Rcpp::List timeSeries, 
	size_t jobs
) {
	return TProto::Translate<TDoubleMatrix>(
		TSpikework::ClassKernelRun(
			TProto::TranslateBack<NDnnProto::TKernelConfig>(kernelConfig), 
			TProto::TranslateBack<TTimeSeries>(timeSeries), 
			jobs
		)
	);
}


// [[Rcpp::export(name = "pp.kernel.run")]]
Rcpp::List PpKernelRun(
	Rcpp::List preProcConfig, 
	Rcpp::List kernelConfig, 
	Rcpp::List timeSeries, 
	size_t jobs,
	double dt = 1.0,
	bool timeCorrelation = false
) {
	TTimeSeries ts;
	TString type = timeSeries.attr("class");
	if (type == "SpikesList") {
		ts = TProto::TranslateBack<TSpikesList>(timeSeries).ConvertToBinaryTimeSeries(dt);
	} else {
		ts = TProto::TranslateBack<TTimeSeries>(timeSeries); 
	}
	auto mVec = TSpikework::KernelRun(
		TProto::TranslateBack<NDnnProto::TPreprocessorConfig>(preProcConfig), 
		TProto::TranslateBack<NDnnProto::TKernelConfig>(kernelConfig), 
		ts, 
		jobs,
		timeCorrelation);

	Rcpp::List ans;
	for (const auto& m: mVec) {
		ans.push_back(TProto::Translate<TDoubleMatrix>(m));
	}

	return ans;
}


// [[Rcpp::export(name = "kernel.run")]]
Rcpp::List KernelRun(
	Rcpp::List kernelConfig, 
	Rcpp::List timeSeries, 
	size_t jobs,
	bool timeCorrelation) 
{
	auto mVec = TSpikework::KernelRun(
		TProto::TranslateBack<NDnnProto::TKernelConfig>(kernelConfig), 
		TProto::TranslateBack<TTimeSeries>(timeSeries), 
		jobs,
		timeCorrelation);

	Rcpp::List ans;
	for (const auto& m: mVec) {
		ans.push_back(TProto::Translate<TDoubleMatrix>(m));
	}

	return ans;
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

// [[Rcpp::export(name = "spike.distance")]]
double SpikeDistance(
	Rcpp::List kernelConfig, 
	Rcpp::List timeSeriesLeft,
	Rcpp::List timeSeriesRight, 
	size_t jobs)
{
	return TSpikework::Distance(
		TProto::TranslateBack<NDnnProto::TKernelConfig>(kernelConfig), 
		TProto::TranslateBack<TTimeSeries>(timeSeriesLeft), 
		TProto::TranslateBack<TTimeSeries>(timeSeriesRight), 
		jobs
	);
}
	
// [[Rcpp::export(name = "pp.spike.distance")]]
double PpSpikeDistance(
	Rcpp::List preProcConfig,
	Rcpp::List kernelConfig, 
	Rcpp::List timeSeriesLeft,
	Rcpp::List timeSeriesRight, 
	size_t jobs,
	double dt = 1.0)
{
	TTimeSeries tsLeft;
	TString typeLeft = timeSeriesLeft.attr("class");
	if (typeLeft == "SpikesList") {
		tsLeft = TProto::TranslateBack<TSpikesList>(timeSeriesLeft).ConvertToBinaryTimeSeries(dt);
	} else {
		tsLeft = TProto::TranslateBack<TTimeSeries>(timeSeriesLeft); 
	}
	TTimeSeries tsRight;
	TString typeRight = timeSeriesRight.attr("class");
	if (typeRight == "SpikesList") {
		tsRight = TProto::TranslateBack<TSpikesList>(timeSeriesRight).ConvertToBinaryTimeSeries(dt);
	} else {
		tsRight = TProto::TranslateBack<TTimeSeries>(timeSeriesRight); 
	}
	return TSpikework::Distance(
		TProto::TranslateBack<NDnnProto::TPreprocessorConfig>(preProcConfig), 
		TProto::TranslateBack<NDnnProto::TKernelConfig>(kernelConfig), 
		std::move(tsLeft),
		std::move(tsRight),
		jobs
	);
}




#endif