#pragma once

#include <dnn/base/base.h>

namespace NDnn {

    template <typename T>
    class TMaybe {
    public:
        TMaybe()
            : Presence(false)
        {
        }
        TMaybe(const T& val)
            : Val(val), Presence(true)
        {
        }

        void operator = (T v) {
            Val = v;
            Presence = true;
        }

        operator bool () const {
            return Defined();
        }

        const T& GetRef() const {
            ENSURE(Presence, "Trying to get maybe that is not set");
            return Val;
        }

        bool Defined() const {
            return Presence;
        }
        
    private:
        T Val;
        bool Presence;
    };

    template <typename T>
    TMaybe<T> Nothing(){
        return TMaybe<T>();
    }

    template <typename T>
    TMaybe<T> Some(T v){
        return TMaybe<T>(v);
    }


} // namespace NDnn