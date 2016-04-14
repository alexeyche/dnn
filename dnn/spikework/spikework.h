#pragma once

#include <dnn/util/matrix.h>
#include <dnn/util/thread.h>

#include <dnn/spikework/kernel/dot.h>
#include <dnn/spikework/kernel/epsp.h>

#include <dnn/spikework/protos/spikework_config.pb.h>

namespace NDnn {

	class TSpikework {
	public:

		static SPtr<IKernel> CreateKernel(const NDnnProto::TKernelConfig& kernelConfig) {
			SPtr<IKernel> kernel;
			if (kernelConfig.has_dot()) {
				ENSURE(!kernel, "Need to choose one kernel");
				kernel = MakeShared(new TDotKernel());
			}
			ENSURE(kernel, "Kernel is not set in config");
			kernel->Deserialize(kernelConfig);
			return kernel;
		}

		static SPtr<IPreprocessor> CreatePreprocessor(const NDnnProto::TPreprocessorConfig& prepProcConfig) {
			SPtr<IPreprocessor> preprocessor;
			if (prepProcConfig.has_epsp()) {
				ENSURE(!preprocessor, "Need to choose one preprocessor");
				preprocessor = MakeShared(new TEpspFilter());
			}
			ENSURE(preprocessor, "Preprocessor is not set in config");
			preprocessor->Deserialize(prepProcConfig);
			return preprocessor;
		}

		static TTimeSeries PreprocessRun(const NDnnProto::TPreprocessorConfig& preProcConfig, TTimeSeries ts, ui32 jobs = 8) {
			SPtr<IPreprocessor> preprocessor = CreatePreprocessor(preProcConfig);
			return preprocessor->Preprocess(std::move(ts));
		}
		
		static TDoubleMatrix KernelRun(
			const NDnnProto::TKernelConfig& kernelConfig, 
			const TTimeSeries& ts, 
			ui32 jobs = 8
		) {
			SPtr<IKernel> kernel = CreateKernel(kernelConfig);

			TVector<TTimeSeries> choppedTs = ts.Chop();
			ENSURE(choppedTs.size()>0, "Got zero sized time series list, check presence of time series information");

			TDoubleMatrix gram(choppedTs.size(), choppedTs.size());

			typedef std::tuple<ui32, ui32, const TTimeSeries&, const TTimeSeries&> TKernCorpus;
			
			TVector<TKernCorpus> corpus;

	        L_DEBUG << "GramMatrix, Calculating kernel values in " << jobs << " jobs";

	        for (ui32 i=0; i<choppedTs.size(); ++i) {
	            for(ui32 j=i; j<choppedTs.size(); ++j) {
	                corpus.push_back(TKernCorpus(i, j, choppedTs[i], choppedTs[j]));
	            }
	        }

	        TVector<std::thread> workers;
	        auto slices = DispatchOnThreads(corpus.size(), jobs);
	        for (const auto& slice: slices) {
	            workers.emplace_back(
	                [&](ui32 from, ui32 to) {
	                    L_DEBUG << "GramMatrix, Working on slice " << from << ":" << to;
	                    for(ui32 iter=from; iter<to; ++iter) {
	                        const auto& tup = corpus[iter];
	                        gram(std::get<0>(tup), std::get<1>(tup)) = kernel->Similarity(std::get<2>(tup), std::get<3>(tup));
	                    }
	                    L_DEBUG << "GramMatrix, " << from << ":" << to << " is done";
	                }, slice.From, slice.To
	            );
	        }
	        for (auto& w: workers) {
	            w.join();
	        }

	        for (ui32 i=0; i<gram.RowSize(); ++i) {
	            gram.SetRowLabel(i, choppedTs[i].GetLabel());
	            for(ui32 j=0; j<i; ++j) {
	                gram(i, j) = gram(j, i);
	            }
	            for(ui32 j=i; j<choppedTs.size(); ++j) {
	                if(i == 0) {
	                    gram.SetColLabel(j, choppedTs[j].GetLabel());
	                }
	            }
	        }
	        return gram;
		}
		
		static TDoubleMatrix KernelRun(
			const NDnnProto::TPreprocessorConfig& preProcConfig, 
			const NDnnProto::TKernelConfig& kernelConfig, 
			TTimeSeries ts, 
			ui32 jobs = 8
		) {
			ts = PreprocessRun(preProcConfig, ts, jobs);
			return KernelRun(kernelConfig, ts, jobs);
		}

	};




} // namespace NDnn