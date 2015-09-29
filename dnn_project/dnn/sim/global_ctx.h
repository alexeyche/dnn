#pragma once

#include <dnn/core.h>
#include <dnn/util/ptr.h>
#include <dnn/sim/sim_info.h>
#include <dnn/base/constants.h>

namespace dnn {


class GlobalCtx {
public:
	GlobalCtx() {}

	void init(const SimInfo &_si, const Constants &_c) {
		si.set(&_si);
		c.set(&_c);
	}

	const SimInfo& getSimInfo() const {
		return si.ref();
	}
	const Constants& getConstants() const {
		return c.ref();
	}
	static GlobalCtx& inst();
private:
	Ptr<const SimInfo> si;
	Ptr<const Constants> c;
};


}