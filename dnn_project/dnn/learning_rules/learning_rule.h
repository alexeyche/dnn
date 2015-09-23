#pragma once

#include <dnn/io/serialize.h>
#include <dnn/util/statistics.h>
#include <dnn/neurons/spike_neuron.h>
#include <dnn/util/ptr.h>
#include <dnn/util/interfaced_ptr.h>
#include <dnn/weight_normalizations/weight_normalization.h>
#include <dnn/protos/learning_rule.pb.h>

namespace dnn {


struct LearningRuleInterface {
	propSynSpikeDelegate propagateSynapseSpike;
	calculateDynamicsDelegate calculateDynamics;
	calculateDynamicsDelegate calculateDynamicsInternal;
};


class Network;
class Builder;

class LearningRuleBase : public SerializableBase {
friend class Network;
friend class Builder;
public:
	typedef LearningRuleInterface interface;

	static void __calculateDynamicsDefault(const Time &t) {}
	static void __propagateSynapseSpikeDefault(const SynSpike &s) {}

	static void provideDefaultInterface(LearningRuleInterface &i) {
		i.calculateDynamics = &LearningRuleBase::__calculateDynamicsDefault;
		i.propagateSynapseSpike =  &LearningRuleBase::__propagateSynapseSpikeDefault;
		i.calculateDynamicsInternal = &LearningRuleBase::__calculateDynamicsDefault;
	}

	template <typename T>
	void provideInterface(LearningRuleInterface &i) {
        i.calculateDynamics = MakeDelegate(static_cast<T*>(this), &T::calculateDynamics);
        i.propagateSynapseSpike = MakeDelegate(static_cast<T*>(this), &T::propagateSynapseSpike);
        i.calculateDynamicsInternal = MakeDelegate(static_cast<T*>(this), &T::calculateDynamicsInternal);
        ifc = i;
    }

	virtual void propagateSynapseSpike(const SynSpike &s) = 0;
	virtual void calculateDynamics(const Time &t) = 0;
	virtual void reset() = 0;

	Statistics& getStat() {
		return stat;
	}
	void calculateDynamicsInternal(const Time& t) {
		ifc.calculateDynamics(t);
		norm.ifc().calculateDynamics(t);
	}

	virtual void setWeightNormalization(Ptr<WeightNormalizationBase> _norm) = 0;

	InterfacedPtr<WeightNormalizationBase>& getWeightNormalization() {
		return norm;
	}

	virtual void linkWithNeuron(SpikeNeuronBase &_n) = 0;
protected:
	Statistics stat;
	InterfacedPtr<WeightNormalizationBase> norm;
	LearningRuleInterface ifc;
};

/*@GENERATE_PROTO@*/
struct LearningRuleInfo : public Serializable<Protos::LearningRuleInfo> {
	void serial_process() {
		begin() << "weight_normalization_is_set: " << weight_normalization_is_set << Self::end;
	}
	bool weight_normalization_is_set;
};

template <typename Constants, typename State, typename Neuron>
class LearningRule : public LearningRuleBase {
public:
	LearningRuleInfo getInfo() {
		LearningRuleInfo info;
		info.weight_normalization_is_set = norm.isSet();
		return info;
	}
	void serial_process() {
		begin() << "Constants: " << c;
		if (messages->size() == 0) {
			(*this) << Self::end;
			return;
		}
		(*this) << "State: " << s << Self::end;
		LearningRuleInfo info;
		if (mode == ProcessingOutput) {
			info = getInfo();
		}

		(*this) << "LearningRuleInfo: "   << info;
		if(info.weight_normalization_is_set) {
			(*this) << "WeightNormalization: " << norm;
		}
	}

	void linkWithNeuron(SpikeNeuronBase &_n) {
		try {
			Neuron &nc = static_cast<Neuron&>(_n);
			n.set(nc);
			if (norm.isSet()) {
				norm.ref().linkWithNeuron(n.ref());
			}
		} catch(std::bad_cast exp) {
			throw dnnException() << "Can't cast neuron for learning rule " << name() << "\n";
		}
	}

	void setWeightNormalization(Ptr<WeightNormalizationBase> _norm) {
		norm = _norm;
		if(n.isSet()) {
			norm.ref().linkWithNeuron(n.ref());
		}
	}
protected:
	Ptr<Neuron> n;

	State s;
	Constants c;
};




}

