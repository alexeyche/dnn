#pragma once

#include <vector>

using std::vector;

#include <dnn/util/pretty_print.h>
#include <dnn/protos/time_series_info.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/util/maybe.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct TimeSeriesInfo : public Serializable<Protos::TimeSeriesInfo> {

    TimeSeriesInfo(): dt(1.0), __current_ahead_position(1) {}
    void serial_process() {
        begin() << "unique_labels: "   << unique_labels   << ", "
                << "labels_start: " << labels_start << ", "
                << "dt: " << dt << Self::end;
    }

    void addLabelAtPos(const string &lab, size_t pos, size_t dur) {
        Maybe<size_t> lab_id;
        auto ulabPtr = unique_labels.begin();
        for(; ulabPtr != unique_labels.end(); ++ulabPtr) {
            if(ulabPtr->first == lab) {
                if(ulabPtr->second != dur) {
                    throw dnnException() << "Trying to add label with same name but another duration\n";
                }
                lab_id = ulabPtr - unique_labels.begin();
            }
        }
        if(!lab_id) {
            unique_labels.push_back(make_pair(lab, dur));
            lab_id = unique_labels.size()-1;
        }
        labels_start.push_back(make_pair(lab_id.getRef(), pos));
    }

    void changeTimeDelta(const double &_dt) {
        dt = _dt;
        for(auto &lt: unique_labels) {
            lt.second = ceil(lt.second/dt);
        }
        for(auto &lt: labels_start) {
            lt.second = ceil(lt.second/dt);
        }
    }

    bool operator == (const TimeSeriesInfo &l) {
        if(labels_start != labels_start) return false;
        if(unique_labels != unique_labels) return false;
        return true;
    }
    bool operator != (const TimeSeriesInfo &l) {
        return ! (*this == l);
    }

    Maybe<size_t> getClassIdFromPosition(double t) {
        if(
           (labels_start[__current_ahead_position-1].second +
            unique_labels[ labels_start[__current_ahead_position-1].first ].second)
           > t
        ) {
            return labels_start[__current_ahead_position-1].first;
        } else {
            return Nothing<size_t>();
        }
    }

    Maybe<size_t> getClassId(double t) {
        while(__current_ahead_position <= labels_start.size()) {
            if(t <= labels_start[__current_ahead_position].second) {
                return getClassIdFromPosition(t);
            }
            __current_ahead_position += 1;
        }
        return Nothing<size_t>();
    }

    vector<pair<string, size_t>> unique_labels; // (label name, duration)
    vector<pair<size_t, size_t>> labels_start; // (label id, time)

    double dt;

    size_t __current_ahead_position;
};


}