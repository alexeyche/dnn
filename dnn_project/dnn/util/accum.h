#pragma once

#include <vector>


namespace dnn {


template <typename T>
class Accum {
public:
    Accum() {}

    Accum(const T &v) {
        acc.push_back(v);
    }
    void operator = (const T &v) {
        acc.push_back(v);
    }

    void operator = (const Accum<T> &v) {
        acc.insert(acc.end(), v.acc.begin(), v.acc.end());
    }

    template <typename NT>
    void operator = (const NT &v) {
        throw dnnException() << "Trying to accumulate value with different type " << v << "\n";
    }

    const vector<T>& getValues() {
        return acc;
    }

private:
    vector<T> acc;

};





}
