#pragma once

#include <dnn/util/ts/time_series.h>

namespace NDnn {

	TTimeSeries Convolve(TTimeSeries& input, TTimeSeries& filter);

} // namespace NDnn