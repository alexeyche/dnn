#include "shoe.h"

#include <dnn/spikework/spikework.h>

namespace NDnn {

	void TShoeKernel::SerialProcess(TProtoSerial& serial) {
        serial(Options, NDnnProto::TKernelConfig::kShoeFieldNumber);
        if (serial.IsInput()) {
            Kernel = TSpikework::CreateKernel(Options.Kernel);      
        }
    }

	double TShoeKernel::PointSimilarity(const TVector<double>& x, const TVector<double>& y) const {
		return std::exp( - Options.Sigma * (Kernel->PointSimilarity(x, x) - 2 * Kernel->PointSimilarity(x, y) + Kernel->PointSimilarity(y, y)));
	}

} // namespace NDnn