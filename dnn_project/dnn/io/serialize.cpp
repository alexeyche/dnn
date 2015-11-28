
#include "serialize.h"

#include <dnn/io/stream.h>
#include <dnn/base/factory.h>

namespace dnn {

void protobinSave(SerializableBase *b, const string fname) {
    std::ofstream f(fname);
    Stream s(f, Stream::Binary);
    s.writeObject(b);
}

Ptr<SerializableBase> SerializableBase::createObject(string name) {
    return Factory::inst().createObject(name);
}


}