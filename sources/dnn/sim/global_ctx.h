#pragma once

#include <dnn/core.h>
#include <dnn/util/ptr.h>

namespace dnn {

class SimInfo;
class Constants;

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