#pragma once

#include <dnn/io/serialize.h>
#include <dnn/neurons/spike_neuron.h>

namespace dnn {

struct ConnectionRecipe {
	ConnectionRecipe() : inhibitory(false), amplitude(1.0), exists(false) {}
	bool inhibitory;
	double amplitude;
	bool exists;
};

class ConnectionBase : public SerializableBase {
public:
	ConnectionBase() {}

	virtual ConnectionRecipe getConnectionRecipe(const SpikeNeuronBase &left, const SpikeNeuronBase &right) = 0;
	void setPreLayerSize(size_t pre_layer_size) {
		_pre_layer_size = pre_layer_size;
	}
	void setPostLayerSize(size_t post_layer_size) {		
		_post_layer_size = post_layer_size;
	}
	const size_t& getPreLayerSize() const {
		return _pre_layer_size;
	}
	const size_t& getPostLayerSize() const {
		return _post_layer_size;
	}
	
protected:
	size_t _pre_layer_size;
	size_t _post_layer_size;
};

template <typename Constants>
class Connection : public ConnectionBase {
	void serial_process() {
		begin() << "Constants: " << c << Self::end;
	}
protected:
	Constants c;
};



}