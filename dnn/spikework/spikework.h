#pragma once

#include <ground/matrix.h>
#include <ground/thread.h>

#include <dnn/spikework/kernel/kernel.h>
#include <dnn/spikework/protos/spikework_config.pb.h>

namespace NDnn {
	using namespace NGround;
	
	class TSpikework {
	public:

		static SPtr<IKernel> CreateKernel(const NDnnProto::TKernelConfig& kernelConfig);

		static SPtr<IPreprocessor> CreatePreprocessor(const NDnnProto::TPreprocessorConfig& prepProcConfig);

		static TTimeSeries PreprocessRun(const NDnnProto::TPreprocessorConfig& preProcConfig, TTimeSeries ts, ui32 jobs = 8) {
			SPtr<IPreprocessor> preprocessor = CreatePreprocessor(preProcConfig);
			return preprocessor->Preprocess(std::move(ts));
		}
		
		static TDoubleMatrix ClassKernelRun(
			const NDnnProto::TKernelConfig& kernelConfig, 
			const TTimeSeries& ts, 
			ui32 jobs = 8) 
		{
			SPtr<IKernel> kernel = CreateKernel(kernelConfig);

			TVector<TTimeSeries> choppedTs = ts.Chop();
			ENSURE(choppedTs.size()>0, "Got zero sized time series list, check presence of time series information");

			ui32 maxLen = 0;
			for (const auto& subTs: choppedTs) {
				maxLen = std::max(maxLen, subTs.Length());
			}
			for (auto& subTs: choppedTs) {
				subTs.PadRightWithZeros(maxLen - subTs.Length());
			}

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
		
		static TDoubleMatrix ClassKernelRun(
			const NDnnProto::TPreprocessorConfig& preProcConfig, 
			const NDnnProto::TKernelConfig& kernelConfig, 
			TTimeSeries ts, 
			ui32 jobs = 8) 
		{
			ts = PreprocessRun(preProcConfig, ts, jobs);
			return ClassKernelRun(kernelConfig, ts, jobs);
		}

		static TVector<TDoubleMatrix> KernelRun(
			const NDnnProto::TKernelConfig& kernelConfig, 
			const TTimeSeries& ts, 
			ui32 jobs = 8,
			bool timeCorrelation = false) 
		{
			SPtr<IKernel> kernel = CreateKernel(kernelConfig);
			TVector<TTimeSeries> choppedTs = ts.Chop();
			if (choppedTs.size() == 0) {
				return { KernelRun(kernel, ts, jobs, timeCorrelation) };
			} else {
				TVector<TDoubleMatrix> ans;
				for (const auto& subTs: choppedTs) {
					ans.emplace_back(KernelRun(kernel, subTs, jobs, timeCorrelation));
				}
				return ans;
			}

		}

		static TVector<TDoubleMatrix> KernelRun(
			const NDnnProto::TPreprocessorConfig& preProcConfig, 
			const NDnnProto::TKernelConfig& kernelConfig, 
			TTimeSeries ts, 
			ui32 jobs = 8,
			bool timeCorrelation = false) 
		{
			ts = PreprocessRun(preProcConfig, ts, jobs);
			return KernelRun(kernelConfig, ts, jobs, timeCorrelation);
		}

		static double Distance(
			const NDnnProto::TPreprocessorConfig& preProcConfig, 
			const NDnnProto::TKernelConfig& kernelConfig, 
			TTimeSeries&& tsLeft,
			TTimeSeries&& tsRight,
			ui32 jobs = 8)
		{
			tsLeft = PreprocessRun(preProcConfig, tsLeft, jobs);
			tsRight = PreprocessRun(preProcConfig, tsRight, jobs);
			return Distance(kernelConfig, std::move(tsLeft), std::move(tsRight), jobs);		
		}

		static double Distance(
			const NDnnProto::TKernelConfig& kernelConfig, 
			TTimeSeries&& tsLeft,
			TTimeSeries&& tsRight,
			ui32 jobs = 8)
		{
			ENSURE(tsLeft.Dim() == tsRight.Dim(), "Need time series with same dimensions");
			int diffLen = tsLeft.Length() - tsRight.Length();
			if (diffLen > 0) {
				tsRight.PadRightWithZeros(diffLen);
			} else 
			if (diffLen < 0) {
				tsLeft.PadRightWithZeros(std::abs(diffLen));
			}
			
			SPtr<IKernel> kernel = CreateKernel(kernelConfig);
			TVector<TIndexSlice> slices;
			if (jobs > tsLeft.Dim()) {
				for (ui32 dimId=0; dimId < tsLeft.Dim(); ++dimId) {
					slices.emplace_back(dimId, dimId + 1);
				}  
			} else {
				slices = DispatchOnThreads(tsLeft.Dim(), jobs);
			}
	        
			TVector<double> ans(tsLeft.Dim(), 0.0);
			TVector<std::thread> workers;
	        for (const auto& slice: slices) {
	        	workers.emplace_back(
	            	[&](ui32 from, ui32 to) {
	                    L_DEBUG << "Distance, Working on dimension " << from << ":" << to << " ...";
    		            for(ui32 iter=from; iter<to; ++iter) {        
		            		ans[iter] = kernel->PointSimilarity(tsLeft.GetVector(iter), tsRight.GetVector(iter));	
		            	}
	            		L_DEBUG << "Distance, Working on dimension " << from << ":" << to << " ... Done";
	                }, slice.From, slice.To
	            );
	       	}
	       	for (auto& w: workers) {
	            w.join();
	        }
	        
	       	return std::accumulate(ans.begin(), ans.end(), 0.0)/tsLeft.Dim(); 		
		} 
			

	private:

		static TDoubleMatrix KernelRun(
			SPtr<IKernel> kernel, 
			const TTimeSeries& ts, 
			ui32 jobs = 8,
			bool timeCorrelation = false)
		{
			if (timeCorrelation) {
				return TimeCorrelationKernelRun(kernel, ts, jobs);
			} else {
				return UnitCorrelationKernelRun(kernel, ts, jobs);
			}
		}

		static TDoubleMatrix TimeCorrelationKernelRun(
			SPtr<IKernel> kernel, 
			const TTimeSeries& ts, 
			ui32 jobs = 8)
		{
			TDoubleMatrix gram(ts.Length(), ts.Length());

			typedef std::tuple<ui32, ui32, const TVector<double>, const TVector<double>> TKernCorpus;

			TVector<TKernCorpus> corpus;

	        L_DEBUG << "GramMatrix, Calculating kernel values in " << jobs << " jobs";

			for (ui32 i=0; i<ts.Length(); ++i) {
	            for(ui32 j=i; j<ts.Length(); ++j) {
	                corpus.push_back(TKernCorpus(i, j, ts.GetColumnVector(i), ts.GetColumnVector(j)));
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
	                        gram(std::get<0>(tup), std::get<1>(tup)) = kernel->PointSimilarity(std::get<2>(tup), std::get<3>(tup));
	                    }
	                    L_DEBUG << "GramMatrix, " << from << ":" << to << " is done";
	                }, slice.From, slice.To
	            );
	        }

	        for (auto& w: workers) {
	            w.join();
	        }
	        
	        for (ui32 i=0; i<gram.RowSize(); ++i) {
	            for(ui32 j=0; j<i; ++j) {
	                gram(i, j) = gram(j, i);
	            }
	        }
	        return gram;

		}

		static TDoubleMatrix UnitCorrelationKernelRun(
			SPtr<IKernel> kernel, 
			const TTimeSeries& ts, 
			ui32 jobs = 8)
		{
			TDoubleMatrix gram(ts.Dim(), ts.Dim());

			typedef std::tuple<ui32, ui32, const TVector<double>&, const TVector<double>&> TKernCorpus;

			TVector<TKernCorpus> corpus;

	        L_DEBUG << "GramMatrix, Calculating kernel values in " << jobs << " jobs";

			for (ui32 i=0; i<ts.Dim(); ++i) {
	            for(ui32 j=i; j<ts.Dim(); ++j) {
	                corpus.push_back(TKernCorpus(i, j, ts.GetVector(i), ts.GetVector(j)));
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
	                        gram(std::get<0>(tup), std::get<1>(tup)) = kernel->PointSimilarity(std::get<2>(tup), std::get<3>(tup));
	                    }
	                    L_DEBUG << "GramMatrix, " << from << ":" << to << " is done";
	                }, slice.From, slice.To
	            );
	        }

	        for (auto& w: workers) {
	            w.join();
	        }
	        
	        for (ui32 i=0; i<gram.RowSize(); ++i) {
	            for(ui32 j=0; j<i; ++j) {
	                gram(i, j) = gram(j, i);
	            }
	        }
	        return gram;
		}
	};




} // namespace NDnn