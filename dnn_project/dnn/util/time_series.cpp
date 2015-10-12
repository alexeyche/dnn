#include "time_series.h"

#include <dnn/io/stream.h>
#include <dnn/util/log/log.h>

namespace dnn {

void TimeSeries::readFromFile(const string &filename, const string &format) {
    ifstream f(filename);
    if(!f.is_open()) {
        throw dnnException()<< "Can't open file " << filename << "\n";
    }
    if(format == "ucr-ts") {
        dim_info.size = 1; // Only one dim TS support
        data.resize(dim_info.size);
        string line;
        while (std::getline(f, line)) {
            string lab;
            convertUcrTimeSeriesLine(line, data[0].values, lab);
            info.addLabelAtPos(lab, data[0].values.size());
        }
    } else
    if(format == "protobin") {
        Ptr<TimeSeries> inp_ts = Stream(f, Stream::Binary).readDynamic<TimeSeries>();
        (*this) = inp_ts.ref();
        inp_ts.destroy();
    } else {
        throw dnnException() << "TimeSeries: unknown format " << format << "\n";
    }
}




}