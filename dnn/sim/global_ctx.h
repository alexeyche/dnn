#pragma once

#include "reward_control.h"

#include <dnn/base/base.h>

#include <dnn/util/ptr.h>
#include <dnn/util/maybe.h>

namespace NDnn {

	template <typename ...T>
	class TSim;

	class TGlobalCtx {

	template <typename ...T>
	friend class TSim;

	public:
		static TGlobalCtx& Inst();

		void Init(TRewardControl& rewardControl) {
			RewardControl.Set(rewardControl);
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

	private:
		void SetCurrentClassId(TMaybe<ui32>&& id) {
			CurrentClassId = id;
		}

		TPtr<TRewardControl> RewardControl;

		TMaybe<ui32> CurrentClassId;
	};



} // namespace NDnn