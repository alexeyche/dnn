#pragma once

#include "reward_control.h"

#include <ground/base/base.h>
#include <ground/matrix.h>

#include <ground/ptr.h>
#include <ground/maybe.h>

namespace NDnn {
	using namespace NGround;
	
	template <typename ...T>
	class TSim;


	struct TDestinationInfo {
		ui32 DestNeuronId;
		double SynapseSign;
	};

	class TGlobalCtx {

	template <typename ...T>
	friend class TSim;

	public:
		static TGlobalCtx& Inst();

		void Init(TRewardControl& rewardControl, const TVector<ui32>& sizeOfLayers, const TVector<TVector<TDestinationInfo>>& adjancentNeuronInfo) {
			RewardControl.Set(rewardControl);
			
			ui32 layerId = 0;
			ui32 cumulativeSize = 0;
			for (const auto& layerSize: sizeOfLayers) {
				cumulativeSize += layerSize;
				while (LayerId.size() < cumulativeSize) {
					LayerId.push_back(layerId);
				}
				++layerId;
			}
			LayerSize = sizeOfLayers.size();
			
			CumulativeError.resize(cumulativeSize, 0.0);
			Error.resize(cumulativeSize, 0.0);
			LastTickError.resize(cumulativeSize, 0.0);
			
			AdjacentNeurons = adjancentNeuronInfo;
		}

		const double& GetReward() const {
			return RewardControl->GetReward();
		}

		double GetRewardDelta() const {
			return RewardControl->GetRewardDelta();
		}

		void PropagateReward(double r) {
			assert(RewardControl.IsSet());
			RewardControl->GatherReward(r);
		}

		const TMaybe<ui32>& GetCurrentClassId() const {
			return CurrentClassId;
		}

		const double& GetPastTime() const {
			return PastTime;
		}
		
		const ui32& GetNeuronLayerId(const ui32& globalNeuronId) const {
			return LayerId[globalNeuronId];
		}
		
		const ui32& GetLayerSize() const {
			return LayerSize;
		}

		void SetError(ui32 lastLayerNeuronId, double error) {
			Error[lastLayerNeuronId] = error;
			CumulativeError[lastLayerNeuronId] += error * error;
		}

		void SetCumulativeError(ui32 globalNeuronId, double error) {
			CumulativeError[globalNeuronId] += error;
		}

		const TVector<double>& GetCumulativeError() const {
			return CumulativeError;
		}
		
		const TVector<double>& GetError() const {
			return LastTickError;
		}
		
		const double& GetError(ui32 lastLayerNeuronId) const {
			return LastTickError[lastLayerNeuronId];
		}
		
		void SwapErrors() {
			LastTickError.swap(Error);
		}

		void SetConnectionInfo(ui32 from, ui32 to, double sign) {
			AdjacentNeurons[from].push_back(TDestinationInfo{to, sign});
		}

		TVector<double> GetCausedErrors(ui32 globalNeuronId) {
			TVector<double> errors;
			for (const auto& adj: AdjacentNeurons[globalNeuronId]) {
				errors.push_back(LastTickError[adj.DestNeuronId]);
			}
			return errors;
		} 

		TVector<double> GetConnectionSign(ui32 from) const {
			TVector<double> signs;
			for (const auto& adj: AdjacentNeurons[from]) {
				signs.push_back(adj.SynapseSign);
			}
			return signs;
		}

		const TVector<TVector<TDestinationInfo>>& GetAdjacentNeuronInfo() const {
			return AdjacentNeurons;
		}

		void ClearConnectionInfo() {
			for (auto& info: AdjacentNeurons) {
				info.clear();
			}
		}

	private:
		void SetPastTime(double pastTime) {
			PastTime = pastTime;
		}
		
		void SetCurrentClassId(TMaybe<ui32>&& id) {
			CurrentClassId = id;
		}

		TPtr<TRewardControl> RewardControl;

		TMaybe<ui32> CurrentClassId;
		double PastTime = 0;
		TVector<ui32> LayerId;
		
		TVector<double> CumulativeError;
		TVector<double> Error;
		TVector<double> LastTickError;
		TVector<TVector<TDestinationInfo>> AdjacentNeurons;  
		ui32 LayerSize;
	};



} // namespace NDnn