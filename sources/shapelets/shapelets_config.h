#pragma once

#include <dnn/io/serialize.h>
#include <shapelets/protos/shapelets_algo.pb.h>


namespace dnn {
namespace shapelets {


/*@GENERATE_PROTO@*/
struct ShapeletsConfig : public Serializable<Protos::ShapeletsConfig> {
    ShapeletsConfig()
    : startSize(10)
    , endSize(100)
    , stepSize(5)
    {
    }

    void serial_process() {
        begin() << "startSize: " << startSize << ", "
                << "endSize: " << endSize << ", "
                << "stepSize: " << stepSize << Self::end;
    }


    size_t startSize;
    size_t endSize;
    size_t stepSize;
};


}
}