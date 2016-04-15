#pragma once

#include <dnn/util/ts/time_series.h>

namespace NDnn {

	class TLyapunov {
	public:

		static TTimeSeries CalculateMetrics(const TTimeSeries& input, const TTimeSeries& net);

	};

} // namespace NDnn