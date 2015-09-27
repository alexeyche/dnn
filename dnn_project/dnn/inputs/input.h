#pragma once

#include <dnn/io/serialize.h>
#include <dnn/util/time_series.h>
#include <dnn/protos/input.pb.h>

namespace dnn {


struct InputInterface {
    retRefDoubleAtTimeDelegate getValue;
};

class InputBase : public SerializableBase {
public:
    typedef InputInterface interface;


    virtual const double& getValue(const Time &t) = 0;
    const size_t& localId() const {
        return _localId;
    }
    virtual void reset() {}
    
    void setLocalId(size_t localId) {
        _localId = localId;
    }

    template <typename T>
    void provideInterface(InputInterface &i) {
        i.getValue = MakeDelegate(static_cast<T*>(this), &T::getValue);
    }
    static const double def_value;
    static const double& getValueDefault(const Time &t) {
        return def_value;
    }

    static void provideDefaultInterface(InputInterface &i) {
        i.getValue = &InputBase::getValueDefault;
    }
private:
    size_t _localId;
};



/*@GENERATE_PROTO@*/
struct InputInfo : public Serializable<Protos::InputInfo> {
    InputInfo() : localId(0) {}

    void serial_process() {
        begin() << "localId: " << localId << Self::end;
    }

    size_t localId;
};

template <typename Constants, typename State>
class Input : public InputBase {
public:
    void serial_process() {
        begin() << "Constants: " << c << ", ";

        if (messages->size() == 0) {
            (*this) << Self::end;
            return;
        }
        InputInfo info;
        if (mode == ProcessingOutput) {
            info.localId = localId();
        }
        (*this) << "InputInfo: " << info;
        (*this) << "State: " << s << Self::end;
    }

protected:
    Constants c;
    State s;
};


}
