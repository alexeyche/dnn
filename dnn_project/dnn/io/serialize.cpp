
#include "serialize.h"

#include <dnn/io/stream.h>

namespace dnn {

void protobinSave(SerializableBase *b, const string fname) {
    std::ofstream f(fname);
    Stream s(f, Stream::Binary);
    s.writeObject(b);
}


}