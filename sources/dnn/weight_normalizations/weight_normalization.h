#pragma once

#include <dnn/io/serialize.h>
#include <dnn/base/exceptions.h>

namespace dnn {


struct WeightNormalizationInterface {
	retDoubleAtRefDoubleDelegate ltp;
	retDoubleAtRefDoubleDelegate ltd;
	retDoubleAtRefDoubleDelegate derivativeModulation;
	calculateDynamicsDelegate calculateDynamics;
};


class WeightNormalizationBase : public SerializableBase {
public:
	typedef WeightNormalizationInterface interface;

	virtual double ltp(const double &w) {
		return 1.0;
	}
    virtual double ltd(const double &w) {
    	return 1.0;
    }
    virtual double derivativeModulation(const double &w) {
    	return 1.0;
    }
    virtual void calculateDynamics(const Time &t) {
    }

    template <typename T>
	void provideInterface(WeightNormalizationInterface &i) {
        i.ltp = MakeDelegate(static_cast<T*>(this), &T::ltp);
        i.ltd = MakeDelegate(static_cast<T*>(this), &T::ltd);
        i.derivativeModulation = MakeDelegate(static_cast<T*>(this), &T::derivativeModulation);
        i.calculateDynamics = MakeDelegate(static_cast<T*>(this), &T::calculateDynamics);
    }


	static double __ltpDefault(const double &w) {
		return 1.0;
	}
	static double __ltdDefault(const double &w) {
		return 1.0;
	}
	static double __derivativeModulationDefault(const double &w) {
		return 1.0;
	}
	static void __calculateDynamicsDefault(const Time &t) {		
	}

	static void provideDefaultInterface(WeightNormalizationInterface &i) {
    	i.ltp = &WeightNormalizationBase::__ltpDefault;
    	i.ltd = &WeightNormalizationBase::__ltdDefault;
    	i.derivativeModulation = &WeightNormalizationBase::__derivativeModulationDefault;
    	i.calculateDynamics = &WeightNormalizationBase::__calculateDynamicsDefault;    	
    }
};

/*@GENERATE_PROTO@*/
struct EmptyState : public Serializable<Protos::EmptyState> {
	void serial_process() {
		begin() << Self::end;
	}
};


template <typename Constants, typename State = EmptyState>
class WeightNormalization : public WeightNormalizationBase {
	void serial_process() {
		begin() << "Constants: " << c;
		if (messages->size() == 0) {
			(*this) << Self::end;
			return;
		}
		if(s.name() != "EmptyState") {
			(*this) << "State: " << s << Self::end;	
		}
		
	}
protected:
	Constants c;
	State s;
};

}
