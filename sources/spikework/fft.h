#pragma once

#include "spikework.h"

namespace dnn {
   
class FFTProcessor : public Processor {
public:
	void process(Spikework::Field &f);
};


}
