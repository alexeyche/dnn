#pragma once

#include <ground/ts/time_series.h>

namespace NDnn {

	class TLyapunov {
	public:

		static NGround::TTimeSeries CalculateMetrics(const NGround::TTimeSeries& input, const NGround::TTimeSeries& net);

	};

} // namespace NDnn