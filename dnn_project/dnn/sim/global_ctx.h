#pragma once

#include <dnn/core.h>
#include <dnn/util/ptr.h>
#include <dnn/util/maybe.h>
#include <dnn/sim/sim_info.h>
#include <dnn/base/constants.h>

#include <dnn/sim/reward_control.h>

namespace dnn {


class GlobalCtx {
friend class Sim;
public:
	GlobalCtx() {}

	void init(const SimInfo &_si, const Constants &_c, double &_duration, RewardControl &_rc) {
		si.set(&_si);
		c.set(&_c);
		rc.set(&_rc);
		duration.set(&_duration);
	}

	const SimInfo& getSimInfo() const {
		return si.ref();
	}
	const Constants& getConstants() const {
		return c.ref();
	}
	static GlobalCtx& inst();

	const Maybe<size_t>& getCurrentClassId() const {
		return _currentClassId;
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
	void setSimDuration(const double d) {
		duration.ref() = std::max(duration.ref(), d);
	}
private:
	void setCurrentClassId(Maybe<size_t> &&id) {
		_currentClassId = id;
	}
	Maybe<size_t> _currentClassId;
	Ptr<const SimInfo> si;
	Ptr<const Constants> c;
	Ptr<RewardControl> rc;
	Ptr<double> duration;
};


}