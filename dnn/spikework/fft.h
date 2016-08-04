#pragma once

#include <ground/ts/time_series.h>
#include <ground/ts/time_series_complex.h>

namespace NDnn {
	using namespace NGround;
	
	class TFFT {
	public:

		static TTimeSeriesComplex Transform(const TTimeSeries& src);

		static TTimeSeries TransformBack(const TTimeSeriesComplex& src);

		static ui32 Nextpow2(ui32 s);
	};

} // namespace NDnn