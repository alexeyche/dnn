#pragma once

#include <vector>


namespace dnn {


template <typename T>
class Accum {
public:
    void operator = (const T &v) {
        acc.push_back(v);
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
