#pragma once

#include <dnn/util/ts/time_series.h>
#include <dnn/util/ts/time_series_complex.h>

namespace NDnn {

	class TFFT {
	public:

		static TTimeSeriesComplex Transform(const TTimeSeries& src);

		static TTimeSeries TransformBack(const TTimeSeriesComplex& src);

		static ui32 Nextpow2(ui32 s);
	};

} // namespace NDnn