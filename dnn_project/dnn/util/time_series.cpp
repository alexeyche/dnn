#include "time_series.h"

#include <dnn/io/stream.h>
#include <dnn/util/log/log.h>

namespace dnn {

void TimeSeries::readFromFile(const string &filename, const string &format) {
    ifstream f(filename);
    if(!f.is_open()) {
        throw dnnException()<< "Can't open file " << filename << "\n";
    }
    if(format == "protobin") {
        Ptr<TimeSeries> inp_ts = Stream(f, Stream::Binary).readDynamic<TimeSeries>();
        (*this) = inp_ts.ref();
        inp_ts.destroy();
    } else {
        throw dnnException() << "TimeSeries: unknown format " << format << "\n";
    }
}




}