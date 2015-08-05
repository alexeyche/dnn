#pragma once

#include <dnn/io/serialize.h>
#include <dnn/util/ptr.h>


namespace dnn {
class TimeSeries;
namespace shapelets {


class Subsequence : public SerializableBase {
public:
    const string name() const {
        return "Subsequence";
    }
    Subsequence() : _from(0), _length(0) {}

    Subsequence(Ptr<TimeSeries> referent, size_t id, size_t from, size_t length)
    : _referent(referent)
    , _id(id)
    , _from(from)
    , _length(length)
    {

    }
    size_t dim() const;

    Ptr<TimeSeries> cutTimeSeries();
    void serial_process();

    const size_t& id() const {
        return _id;
    }
    const size_t& from() const {
        return _from;
    }
    const size_t& length() const {
        return _length;
    }
    Ptr<TimeSeries> referent() {
        return _referent;
    }
private:
    Ptr<TimeSeries> _referent;
    size_t _id;
    size_t _from;
    size_t _length;
};


}
}