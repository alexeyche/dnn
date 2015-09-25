#pragma once

#include <dnn/connections/connection.h>
#include <dnn/base/constants.h>
#include <dnn/io/stream.h>

#include <dnn/util/json.h>


namespace dnn {

using namespace rapidjson;


class Builder {
public:
	struct Layer {
		vector<InterfacedPtr<SpikeNeuronBase>> neurons;
	};



	Builder(const Constants &_c) : c(_c), input_stream(nullptr) {
	}

	vector<InterfacedPtr<SpikeNeuronBase>> buildNeurons();

	static void turnOnStatistics(vector<InterfacedPtr<SpikeNeuronBase>> &neurons, const vector<size_t> &ids) {
		for(auto it=ids.cbegin(); it != ids.cend(); ++it) {
			if(*it >= neurons.size()) {
				throw dnnException() << "Can't find neuron " << *it << " to listen\n";
			}
			neurons[*it].ref().stat.turnOn();
			for(auto s: neurons[*it].ref().getSynapses()) {
				s.ref().stat.turnOn();
			}
			if(neurons[*it].ref().lrule.isSet()) {
				neurons[*it].ref().lrule.ref().stat.turnOn();
				if(neurons[*it].ref().lrule.ref().norm.isSet()) {
					neurons[*it].ref().lrule.ref().norm.ref().stat.turnOn();
				}
			}
		}
	}
	void connectLayers(Layer &pre, Layer &post, const string &conn_conf_s) {
		Document conn_conf = Json::parseStringC(conn_conf_s);
		Ptr<ConnectionBase> conn = buildObjectFromConstants<ConnectionBase>(Json::getStringVal(conn_conf, "type"), c.connections);

		for (auto &npre : pre.neurons) {
			for (auto &npost : post.neurons) {
				if(npre.ref().id() == npost.ref().id()) {
					continue;
				}
				ConnectionRecipe connection_recipe = conn->getConnectionRecipe(npre.ref(), npost.ref());
				if (connection_recipe.exists) {
					Ptr<SynapseBase> syn;
					if(!connection_recipe.inhibitory) {
					    syn = buildObjectFromConstants<SynapseBase>(
					    	Json::getStringVal(conn_conf, "synapse"),
					    	c.synapses
					    );
					} else {
						string inh_synapse_type = Json::getStringValDef(conn_conf, "inh_synapse", "");
						if(inh_synapse_type.empty()) {
							throw dnnException() << "Connection " << conn->name() << " demands inhibitory synapse type to be pointed in constants\n";
						}
					    syn = buildObjectFromConstants<SynapseBase>(
					    	inh_synapse_type,
					    	c.synapses
					    );
					}
					assert(syn);

					syn->mutIdPre() = npre.ref().id();
					syn->mutDendriteDelay() = Json::getDoubleValDef(conn_conf, "dendrite_delay", 0.0);
					syn->mutWeight() = connection_recipe.amplitude * Json::getDoubleVal(conn_conf, "start_weight");
					npost.ref().addSynapse(InterfacedPtr<SynapseBase>(syn.ptr()));
				}
			}
		}
	}

	template <typename T>
	static Ptr<T> buildObjectFromConstants(const string &name, const map<string, string> &object_const_map) {
		auto cptr = object_const_map.find(name);
		if ( cptr == object_const_map.end() ) {
			throw dnnException() << "Trying to build " << name << " from constants and can't find him\n";
		}

		Document const_json = Json::parseStringC(cptr->second);

		Document d;
		Value cv(kObjectType);
		Value copy_v;
		copy_v.CopyFrom(const_json, Json::d.GetAllocator());
		cv.AddMember(StringRef(name.c_str()), copy_v, Json::d.GetAllocator());
		string processed_const = Json::stringify(cv);

		istringstream *ss = new istringstream(processed_const);
		Stream s(*ss, Stream::Text);
		Ptr<T> n = s.read<T>();
		delete ss;
		if(!n) {
			throw dnnException() << "Null object from constants: " << name << "\n";
		}
		return n;
	}

	void setInputModelStream(Stream *s) {
		input_stream = s;
	}


private:
	Stream *input_stream;
	const Constants &c;
};

}