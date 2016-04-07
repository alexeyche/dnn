#include "convolve.h"
#include "fft.h"

#include <dnn/util/ts/time_series_complex.h>

namespace NDnn {

    TTimeSeries Convolve(TTimeSeries& input, TTimeSeries& filter) {
        ui32 paddingSize = filter.Length();

        input.PadRightWithZeros(paddingSize);
        filter.PadRightWithZeros(input.Length()-paddingSize);

        ui32 paddingNfft = TFFT::Nextpow2(input.Length())-input.Length();
        paddingSize += paddingNfft;

        input.PadRightWithZeros(paddingNfft);
        filter.PadRightWithZeros(paddingNfft);

        TTimeSeriesComplex inputFft = TFFT::Transform(input);
        TTimeSeriesComplex filterFft = TFFT::Transform(filter);

        inputFft.InnerProduct(filterFft);

        TTimeSeries output = TFFT::TransformBack(inputFft);
        output.CutFromRight(paddingSize);

        return output;
    }

} // namespace NDnn