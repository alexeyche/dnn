#pragma once

#include <dnn/connections/connection.h>
#include <dnn/base/constants.h>
#include <dnn/io/stream.h>
#include <dnn/sim/reward_control.h>

#include <dnn/util/json.h>


namespace dnn {

using namespace rapidjson;


class Builder {
public:
	struct Layer {
		size_t size() const {
			return neurons.size();
		}
		vector<InterfacedPtr<SpikeNeuronBase>> neurons;		
	};



	Builder(const Constants &_c) : c(_c), input_stream(nullptr) {
	}

	RewardControl buildRewardControlFromConstants();

	vector<InterfacedPtr<SpikeNeuronBase>> buildNeurons();

	static void turnOnStatistics(vector<InterfacedPtr<SpikeNeuronBase>> &neurons) {
		for(auto &n: neurons) {
			n.ref().stat.turnOn();
			for(auto s: n.ref().getSynapses()) {
				s.ref().stat.turnOn();
			}
			if(n.ref().lrule.isSet()) {
				n.ref().lrule.ref().stat.turnOn();

			}
			if(n.ref().norm.isSet()) {
				n.ref().norm.ref().stat.turnOn();
			}
		}
	}
	void connectLayers(Layer &pre, Layer &post, const string &conn_conf_s) {
		Document conn_conf = Json::parseStringC(conn_conf_s);
		Ptr<ConnectionBase> conn = buildObjectFromConstants<ConnectionBase>(Json::getStringVal(conn_conf, "type"), c.connections);

		conn->setPreLayerSize(pre.size());
		conn->setPostLayerSize(post.size());
		
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
			throw dnnException() << "Trying to build " << name << " from constants and can't find it\n";
		}

		Document const_json = Json::parseStringC(cptr->second);

		string processed_const = Json::stringify(Json::makeDocument(name, const_json));

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

	map<string, string> getInputFileNames() const {
	    map<string, string> fnames;
	    for (auto it = c.sim_conf.files.begin(); it != c.sim_conf.files.end(); ++it) {
	        const string &obj_name = it->first;
	        Document file_conf = Json::parseStringC(it->second);

	        string fname = Json::getStringVal(file_conf, "filename");
	        if(strStartsWith(fname, "@")) {
	            continue;
	        }
	        fnames[obj_name] = fname;
	    }
	    return fnames;
	}

	const TimeSeriesInfo& getInputTimeSeriesInfo() const;

private:
	Stream *input_stream;
	const Constants &c;
};

}