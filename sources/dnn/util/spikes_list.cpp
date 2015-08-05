
#include "spikes_list.h"



namespace dnn {


Ptr<TimeSeries> SpikesList::convertToBinaryTimeSeries(const double &dt) const {
    Ptr<TimeSeries> out(Factory::inst().createObject<TimeSeries>());

    out->info = ts_info;

    double max_spike_time = std::numeric_limits<double>::min();
    size_t max_size = std::numeric_limits<size_t>::min();
    for(size_t di=0; di<seq.size(); ++di) {
        double t=0;
        for(const auto&spike_time: seq[di].values) {
            // cout << "dim: " << di << ", t: " << t << ", spike_time: " << spike_time << "\n";
            while(t<spike_time) {
                out->addValue(di, 0.0);
                t+=dt;
                // cout << t << ", ";
            }
            // cout << "\n";
            out->addValue(di, 1.0);
            t+=dt;
            max_spike_time = std::max(max_spike_time, spike_time);
        }
        max_size = std::max(max_size, out->data[di].values.size());
    }
    // cout << "max_size: " << max_size << ", max_spike_time: " << max_spike_time << "\n";

    for(size_t di=0; di<out->data.size(); ++di) {
        double last_t = dt*out->data[di].values.size();
        // cout << "dim: " << di << ", last_t: " << last_t << ", size: " <<  out->data[di].values.size() << "\n";
        while(last_t <=max_spike_time) {
            out->addValue(di, 0.0);
            last_t +=dt;
            // cout << last_t << ", ";
        }
        // cout << "\n";
        if(out->data[di].values.size() != max_size) {
            throw dnnException() << "Failed to convert spike times to " << \
                                    "equal time series for " << di << " dimension, " << \
                                    out->data[di].values.size() << " != max_size " << max_size << " \n";
        }
    }
    return out;
}

}