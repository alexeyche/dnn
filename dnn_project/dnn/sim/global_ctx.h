#pragma once

#include <dnn/core.h>
#include <dnn/util/ptr.h>
#include <dnn/sim/sim_info.h>
#include <dnn/base/constants.h>

#include <dnn/sim/reward_control.h>

namespace dnn {


class GlobalCtx {
friend class Sim;
public:
	GlobalCtx() {}

	void init(const SimInfo &_si, const Constants &_c, RewardControl &_rc) {
		si.set(&_si);
		c.set(&_c);
		rc.set(&_rc);
	}

	const SimInfo& getSimInfo() const {
		return si.ref();
	}
	const Constants& getConstants() const {
		return c.ref();
	}
	static GlobalCtx& inst();

	const size_t& getCurrentClassId() const {
		assert(_currentClassId.isSet());
		return _currentClassId.ref();
	}

	void propagateReward(const double &R) {
		assert(rc.isSet());
		rc->gatherReward(R);
	}
	const double& getReward() const {
		return rc->getReward();
	}
	const double getRewardDelta() const {
		return rc->getRewardDelta();
	}
private:
	void setCurrentClassId(const size_t &id) {
		_currentClassId.set(&id);
	}
	Ptr<const size_t> _currentClassId;
	Ptr<const SimInfo> si;
	Ptr<const Constants> c;
	Ptr<RewardControl> rc;
};


}