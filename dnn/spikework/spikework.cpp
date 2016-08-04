#include "spikework.h"


#include <dnn/spikework/kernel/dot.h>
#include <dnn/spikework/kernel/rbf_dot.h>
#include <dnn/spikework/kernel/anova_dot.h>
#include <dnn/spikework/kernel/epsp.h>
#include <dnn/spikework/kernel/gauss.h>
#include <dnn/spikework/kernel/shoe.h>

namespace NDnn {

	SPtr<IPreprocessor> TSpikework::CreatePreprocessor(const NDnnProto::TPreprocessorConfig& prepProcConfig) {
		SPtr<IPreprocessor> preprocessor;
		if (prepProcConfig.has_epsp()) {
			ENSURE(!preprocessor, "Need to choose one preprocessor");
			preprocessor = MakeShared(new TEpspFilter());
		}
		if (prepProcConfig.has_gauss()) {
			ENSURE(!preprocessor, "Need to choose one preprocessor");
			preprocessor = MakeShared(new TGaussFilter());
		}
		ENSURE(preprocessor, "Preprocessor is not set in config");
		preprocessor->Deserialize(prepProcConfig);
		return preprocessor;
	}


	SPtr<IKernel> TSpikework::CreateKernel(const NDnnProto::TKernelConfig& kernelConfig) {
		SPtr<IKernel> kernel;
		if (kernelConfig.has_dot()) {
			ENSURE(!kernel, "Need to choose one kernel");
			kernel = MakeShared(new TDotKernel());
		}
		if (kernelConfig.has_rbfdot()) {
			ENSURE(!kernel, "Need to choose one kernel");
			kernel = MakeShared(new TRbfDotKernel());
		}
		if (kernelConfig.has_anovadot()) {
			ENSURE(!kernel, "Need to choose one kernel");
			kernel = MakeShared(new TAnovaDotKernel());
		}
		if (kernelConfig.has_shoe()) {
			ENSURE(!kernel, "Need to choose one kernel");
			kernel = MakeShared(new TShoeKernel());
		}
		ENSURE(kernel, "Kernel is not set in config: " << kernelConfig.DebugString());
		kernel->Deserialize(kernelConfig);
		return kernel;
	}

} // namespace NDnn
