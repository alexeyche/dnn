#pragma once

#include "reward_control.h"

#include <ground/base/base.h>

#include <ground/ptr.h>
#include <ground/maybe.h>

namespace NDnn {
	using namespace NGround;
	
	template <typename ...T>
	class TSim;

	class TGlobalCtx {

	template <typename ...T>
	friend class TSim;

	public:
		static TGlobalCtx& Inst();

		void Init(TRewardControl& rewardControl, const TVector<ui32>& sizeOfLayers) {
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
		ui32 LayerSize;
	};



} // namespace NDnn