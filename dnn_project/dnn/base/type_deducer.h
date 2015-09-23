#pragma once


namespace dnn {

class TypeDeducer {
public:
    virtual string deduceType(const std::type_info &info) const = 0;

    virtual ~TypeDeducer() {}
};


}
