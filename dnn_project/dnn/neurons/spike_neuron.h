#pragma once


#include <dnn/base/base.h>
#include <dnn/base/sim_element.h>
#include <dnn/util/interfaced_ptr.h>
#include <dnn/act_functions/act_function.h>
#include <dnn/synapses/synapse.h>
#include <dnn/inputs/input.h>
#include <dnn/io/serialize.h>
#include <dnn/util/statistics.h>
#include <dnn/learning_rules/learning_rule.h>
#include <dnn/util/act_vector.h>
#include <dnn/util/spikes_list.h>
#include <dnn/protos/spike_neuron.pb.h>
#include <dnn/reinforcements/reinforcement.h>
#include <dnn/weight_normalizations/weight_normalization.h>

namespace dnn {


struct SpikeNeuronInterface {
	calculateNeuronDynamicsDelegate calculateDynamics;
	calculateDynamicsDelegate postSpikeDynamics;
};
extern size_t global_neuron_index;

class Builder;
class Network;
class Sim;

class SpikeNeuronBase : public SimElement {
friend class Builder;
friend class Network;
friend class Sim;
public:
	SpikeNeuronBase() : input_queue_lock(ATOMIC_FLAG_INIT), _fired(false), _u(0.0), _p(0.0), _mod(1.0) {
		_id = global_neuron_index++;
	}

	typedef SpikeNeuronInterface interface;

	const size_t& id() const;
	void setCoordinates(size_t xi, size_t yi, size_t colSize);
	const size_t localId() const;
	const size_t& xi() const;
	const size_t& yi() const;
	const size_t& colSize() const;
	const double& axonDelay() const;
	double& mutAxonDelay();
	const bool& fired() const;

	void setFired(const bool& f);

	void setLearningRule(Ptr<LearningRuleBase> _lrule);
	const InterfacedPtr<LearningRuleBase>& getLearningRule() const;

	void setActFunction(Ptr<ActFunctionBase> _act_f);
	const InterfacedPtr<ActFunctionBase>& getActFunction() const;

	void setWeightNormalization(Ptr<WeightNormalizationBase> _norm);
	const InterfacedPtr<WeightNormalizationBase>& getWeightNormalization() const;

	void setReinforcement(Ptr<ReinforcementBase> _reinforce);

	void setInput(Ptr<InputBase> _input);

	bool inputIsSet();
	const InputBase& getInput() const;
	void addSynapse(InterfacedPtr<SynapseBase> syn);
	const ActVector<InterfacedPtr<SynapseBase>>& getSynapses() const;
	ActVector<InterfacedPtr<SynapseBase>>& getMutSynapses();

	template <typename T>
	void provideInterface(SpikeNeuronInterface &i) {
        i.calculateDynamics = MakeDelegate(static_cast<T*>(this), &T::calculateDynamics);
        i.postSpikeDynamics = MakeDelegate(static_cast<T*>(this), &T::postSpikeDynamics);
        ifc = i;
	}
	static void __calculateDynamicsDefault(const Time &t, const double &Iinput, const double &Isyn) {
		throw dnnException()<< "Calling inapropriate default interface function\n";
	}
	static void __postSpikeDynamicDefault(const Time &t) {
		throw dnnException()<< "Calling inapropriate default interface function\n";
	}

	static void provideDefaultInterface(SpikeNeuronInterface &i) {
		i.calculateDynamics = &SpikeNeuronBase::__calculateDynamicsDefault;
		i.postSpikeDynamics =  &SpikeNeuronBase::__postSpikeDynamicDefault;
	}

	void resetInternal();
	void initInternal();

	virtual void calculateDynamics(const Time& t, const double &Iinput, const double &Isyn) = 0;
	virtual void postSpikeDynamics(const Time& t) {}

	Statistics getStat();

	void enqueueSpike(const SynSpike && sp);

	void readInputSpikes(const Time &t);
	void calculateDynamicsInternal(const Time &t);

	const double& getFiringProbability() const {
        return _p;
    }
    const double& getMembrane() const {
        return _u;
    }
    const double& getProbabilityModulation() const {
        return _mod;
    }
    double& membrane() {
    	return _u;
    }
	double& firingProbability() {
    	return _p;
    }
   	double& probabilityModulation() {
   		return _mod;
   	}

protected:
	bool _fired;
	size_t _id;
	size_t _xi;
	size_t _yi;
	size_t _colSize;
	
	double _p;
	double _u;
	double _mod;

	double _axonDelay;
	ActVector<InterfacedPtr<SynapseBase>> syns;

	InterfacedPtr<ActFunctionBase> act_f;
	InterfacedPtr<InputBase> input;
	InterfacedPtr<LearningRuleBase> lrule;
	InterfacedPtr<ReinforcementBase> reinforce;
	InterfacedPtr<WeightNormalizationBase> norm;

	Statistics stat;

	priority_queue<SynSpike> input_spikes;
	std::atomic_flag input_queue_lock;

	SpikeNeuronInterface ifc;
};

/*@GENERATE_PROTO@*/
struct SpikeNeuronInfo : public Serializable<Protos::SpikeNeuronInfo> {
	void serial_process() {
		begin() << "id: " << id << ", "
				<< "p: " << p << ", "
				<< "u: " << u << ", "
				<< "xi: " << xi << ", "
				<< "yi: " << yi << ", "
				<< "colSize: " << colSize << ", "
		        << "axonDelay: " << axonDelay << ", "
		        << "num_of_synapses: " << num_of_synapses << ", "
		        << "act_function_is_set: " << act_function_is_set << ", "
		        << "input_is_set: " << input_is_set << ", "
		        << "lrule_is_set: " << lrule_is_set << ", "
		        << "reinforce_is_set: " << reinforce_is_set  << ", "
		        << "weight_normalization_is_set: " << weight_normalization_is_set << Self::end;
	}
	double u;
	double p;
	size_t id;
	size_t xi;
	size_t yi;
	size_t colSize;
	double axonDelay;
	size_t num_of_synapses;
	bool act_function_is_set;
	bool input_is_set;
	bool lrule_is_set;
	bool reinforce_is_set;
	bool weight_normalization_is_set;
};

template <typename Constants, typename State>
class SpikeNeuron : public SpikeNeuronBase {
public:
	SpikeNeuronInfo getInfo() {
		SpikeNeuronInfo info;
		info.id = id();
		info.u = getMembrane();
		info.p = getFiringProbability();
		info.xi = xi();
		info.yi = yi();
		info.colSize = colSize();
		info.axonDelay = axonDelay();
		info.num_of_synapses = syns.size();
		info.act_function_is_set = act_f.isSet();
		info.input_is_set = input.isSet();
		info.lrule_is_set = lrule.isSet();
		info.reinforce_is_set = reinforce.isSet();
		info.weight_normalization_is_set = norm.isSet();
		return info;
	}

	void serial_process() {
		begin() << "Constants: " << c;

		if (messages->size() == 0) {
			(*this) << Self::end;
			return;
		}

		(*this) << "State: " << s;

		if (messages->size() == 0) {
			(*this) << Self::end;
			return;
		}

		SpikeNeuronInfo info;
		if (mode == ProcessingOutput) {
			info = getInfo();
		}

		(*this) << "SpikeNeuronInfo: "   << info;

		if (info.act_function_is_set) {
			(*this) << "ActFunction: " << act_f;
		}
		if (info.input_is_set) {
			(*this) << "Input: " << input;
		}
		if (info.lrule_is_set) {
			(*this) << "LearningRule: " << lrule;
			lrule.ref().linkWithNeuron(*this);
		}
		if (info.reinforce_is_set) {
			(*this) << "Reinforcement: " << reinforce;
			reinforce.ref().linkWithNeuron(*this);
		}
		if (info.weight_normalization_is_set) {
			(*this) << "WeightNormalization: " << norm;
			norm.ref().linkWithNeuron(*this);
		}
		if(mode == ProcessingInput) {
			syns.resize(info.num_of_synapses);
		}
		for (size_t i = 0; i < info.num_of_synapses; ++i) {
			(*this) << "Synapse: " << syns[i];
		}
		if(mode == ProcessingInput) {
			_xi = info.xi;
			_yi = info.yi;
			_id = info.id;
			_colSize = info.colSize;
			_axonDelay = info.axonDelay;
		}
		(*this) << Self::end;
	}

protected:
	State s;
	Constants c;
};


}
