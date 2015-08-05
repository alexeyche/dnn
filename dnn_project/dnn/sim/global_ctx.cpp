#include "global_ctx.h"

namespace dnn {


GlobalCtx& GlobalCtx::inst() {
	static GlobalCtx _inst;
	return _inst;
}

}
