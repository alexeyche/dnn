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
			SizeOfLayers = sizeOfLayers;
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
		
		ui32 GetNeuronLayerId(ui32 globalNeuronId) {
			ui32 id = 0;
			while (id < SizeOfLayers.size()) {
				if (SizeOfLayers[id] > globalNeuronId) {
					return id;
				}
			}
			throw TErrException() << "Can't find neuron by global id: " << globalNeuronId;
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
		TVector<ui32> SizeOfLayers;
	};



} // namespace NDnn