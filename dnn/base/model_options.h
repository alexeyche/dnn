#pragma once

#include <ground/optional.h>
#include <ground/base/base.h>

namespace NDnn {
	using namespace NGround;
	
	struct TModelOptions {
		TOptional<ui32> Port;
		TOptional<ui32> Seed;
		TOptional<ui32> ConnectionSeed;
		TOptional<ui32> Jobs;
		TOptional<TString> ConfigFile;
		TOptional<TString> ModelLoad;
		TOptional<TString> ModelSave;
		TOptional<TString> InputSpikesFile;
		TOptional<TString> InputTimeSeries;
		TOptional<TString> OutputSpikesFile;
		TOptional<TString> StatFile;
		TOptional<double> Tmax;
		TString Name;
		bool NoLearning = false;
	};

} // namespace NDnn