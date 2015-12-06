#pragma once

namespace dnn {


template <typename T>
class Maybe {
public:
    Maybe()
        : presence(false)
    {
    }
    Maybe(const T& _val)
        : val(_val), presence(true)
    {
    }

    void operator = (T v) {
        val = v;
        presence = true;
    }

    operator bool () {
        return presence;
    }

    const T& getRef() const {
        if(!presence) {
            throw dnnException() << "Trying to get maybe that is not set";
        }
        return val;
    }
private:
    T val;
    bool presence;
};

template <typename T>
Maybe<T> Nothing(){
    return Maybe<T>();
}


}