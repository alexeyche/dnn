#pragma once

#include <dnn/io/serialize.h>

namespace dnn {

class SimElement : public SerializableBase {
public:
    virtual void reset() {}
    virtual void init() {}
};



}