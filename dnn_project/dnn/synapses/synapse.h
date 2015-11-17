#pragma once

#include <dnn/io/serialize.h>
#include <dnn/util/statistics.h>
#include <dnn/protos/synapse.pb.h>
#include <dnn/base/sim_element.h>

namespace dnn {


struct SynapseInterface {
	stateDelegate propagateSpike;
	calculateDynamicsDelegate calculateDynamics;
};

class Network;
class SpikeNeuronBase;
class Builder;

class SynapseBase : public SimElement {
friend class Network;
friend class SpikeNeuronBase;
friend class Builder;
public:
	SynapseBase() : _fired(false), _potential(0.0), _weight(0.0), _dendriteDelay(0.0), _amplitude(0.0) {}
	typedef SynapseInterface interface;

	inline const size_t& idPre() const {
		return _idPre;
	}
	inline size_t& mutIdPre() {
		return _idPre;
	}

	virtual void propagateSpike() {
	    mutPotential() += amplitude();
	}

	virtual void calculateDynamics(const Time &t) = 0;

	template <typename T>
	void provideInterface(SynapseInterface &i) {
        i.propagateSpike = MakeDelegate(static_cast<T*>(this), &T::propagateSpike);
        i.calculateDynamics = MakeDelegate(static_cast<T*>(this), &T::calculateDynamics);
    }

	static void __defaultPropagateSpike() {
		throw dnnException() << "Calling inapropriate default interface function\n";
	}
	static void __defaultCalculateDynamics(const Time &t) {
		throw dnnException() << "Calling inapropriate default interface function\n";
	}

	static void provideDefaultInterface(SynapseInterface &i) {
		i.propagateSpike = &SynapseBase::__defaultPropagateSpike;
        i.calculateDynamics = &SynapseBase::__defaultCalculateDynamics;
	}

	Statistics& getStat() {
		return stat;
	}
	inline void setFired(bool fired) {
		_fired = fired;
	}
	inline const double fired() const {
		return _fired;
	}
	inline double& mutWeight() {
		return _weight;
	}
	inline const double& weight() const {
		return _weight;
	}
	inline const double& dendriteDelay() {
		return _dendriteDelay;
	}
	inline double& mutDendriteDelay() {
		return _dendriteDelay;
	}
	inline const double& potential() const {
		return _potential;
	}
	inline double& mutPotential() {
		return _potential;
	}
	inline double getWeightedPotential() const {
		return weight() * potential();
	}
	inline const double& amplitude() const {
		return _amplitude;
	}
	inline double& mutAmplitude() {
		return _amplitude;
	}
	const bool isExcitatory() const {
		return amplitude()>0;
	}
	const bool isInhibitory() const {
		return !isExcitatory();
	}


protected:
	size_t _idPre;
	double _dendriteDelay;
	double _weight;
	double _potential;
	bool _fired;

	double _amplitude;

	Statistics stat;
};


/*@GENERATE_PROTO@*/
struct SynapseInfo : public Serializable<Protos::SynapseInfo> {
	void serial_process() {
		begin() << "idPre: " 		  << idPre 		<< ", " \
		        << "dendriteDelay: " << dendriteDelay << ", " \
		        << "weight: " 		  << weight 	  << ", " \
		        << "potential: " << potential	<< Self::end;
	}
	size_t idPre;
	double dendriteDelay;
	double weight;
	double potential;
};



template <typename Constants, typename State>
class Synapse : public SynapseBase {

public:
	SynapseInfo getInfo() {
		SynapseInfo info;
		info.idPre = _idPre;
		info.dendriteDelay = _dendriteDelay;
		info.weight = _weight;
		info.potential = _potential;
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
		SynapseInfo info;
		if (mode == ProcessingOutput) {
			info = getInfo();
		}

		(*this) << "SynapseInfo: "   << info;

		if (mode == ProcessingInput) {
			_weight = info.weight;
			_idPre = info.idPre;
			_dendriteDelay = info.dendriteDelay;
			_potential = info.potential;
		}

		(*this) << Self::end;
	}
protected:
	State s;
	Constants c;
};


}
