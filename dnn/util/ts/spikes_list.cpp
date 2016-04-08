#include "spikes_list.h"

#include "time_series.h"

namespace NDnn {

    TTimeSeries TSpikesList::ConvertToBinaryTimeSeries(double dt) const {
        TTimeSeries out;

        out.Info = Info;
        out.SetDimSize(Dim());
        double max_spike_time = std::numeric_limits<double>::min();
        ui32 max_size = std::numeric_limits<ui32>::min();
        for(ui32 di=0; di<Data.size(); ++di) {
            double t=0;
            for(const auto &spike_time: Data[di].Values) {
                // cout << "dim: " << di << ", t: " << t << ", spike_time: " << spike_time << "\n";
                while (t<spike_time) {
                    out.AddValue(di, 0.0);
                    t += dt;
                    // cout << t << ", ";
                }
                // cout << "\n";
                out.AddValue(di, 1.0);
                t += dt;
                if (spike_time > max_spike_time) {
                    max_spike_time = spike_time;
                }
            }
            max_size = std::max(max_size, out.Data[di].Values.size());
        }
        // cout << "max_size: " << max_size << ", max_spike_time: " << max_spike_time << " " << max_id << "\n";

        for(ui32 di=0; di<out.Data.size(); ++di) {
            double last_t = dt * out.Data[di].Values.size();
            // cout << "dim: " << di << ", last_t: " << last_t << ", size: " <<  out->data[di].values.size() << "\n";
            while(last_t <= max_spike_time) {
                out.AddValue(di, 0.0);
                last_t +=dt;
                // cout << last_t << ", ";
            }
            // cout << "\n";
            if(out.Data[di].Values.size() != max_size) {
                throw TErrException() << "Failed to convert spike times to " << \
                                        "equal time series for " << di << " dimension, " << \
                                        out.Data[di].Values.size() << " != max_size " << max_size << " \n";
            }
        }

        return out;
    }

} // namespace NDnnProto