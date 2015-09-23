#include "time_series.h"

#include <dnn/io/stream.h>
#include <dnn/base/factory.h>


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
        delete inp_ts.ptr();
    } else {
        throw dnnException() << "TimeSeries: unknown format " << format << "\n";
    }
}


vector<Ptr<TimeSeries>> TimeSeries::chop()  {
    size_t elem_id = 0;
    vector<Ptr<TimeSeries>> ts_chopped;
    assert(info.labels_timeline.size() == info.labels_ids.size());
    for(size_t li=0; li<info.labels_timeline.size(); ++li) {
        const size_t &end_of_label = info.labels_timeline[li];
        const size_t &label_id = info.labels_ids[li];
        const string &label = info.unique_labels[label_id];

        Ptr<TimeSeries> labeled_ts(Factory::inst().createObject<TimeSeries>());
        for(; elem_id < end_of_label; ++elem_id) {
            for(size_t di=0; di<data.size(); ++di) {
                labeled_ts->addValue(di, data[di].values[elem_id]);
            }
        }
        labeled_ts->info.addLabelAtPos(label, labeled_ts->length());
        ts_chopped.push_back(labeled_ts);
    }
    return ts_chopped;
}

}