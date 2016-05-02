#pragma once

#include <ground/ts/time_series.h>

namespace NDnn {
	using namespace NGround;
	
	TTimeSeries Convolve(TTimeSeries& input, TTimeSeries& filter);

} // namespace NDnn