#pragma once


#include <dnn/contrib/rapidjson/document.h>
#include <dnn/contrib/rapidjson/stringbuffer.h>
#include <dnn/contrib/rapidjson/prettywriter.h>
#include <dnn/contrib/rapidjson/filestream.h>

#include <dnn/util/json.h>
#include <dnn/base/exceptions.h>

#include <dnn/util/util.h>

namespace dnn {

using namespace rapidjson;


struct SimConfiguration : public Printable {
	SimConfiguration() : dt(1.0), seed(-1) {}

	vector<string> layers;
	multimap< pair<size_t, size_t>, string> conn_map;
	double dt;
	int seed;
	vector<size_t> neurons_to_listen;
	map<string, string> files;
	string reward_dynamics;

	void print(ostream &o) const {
		o << "layers: \n";
		for (auto &v : layers) {
			o << "  " <<  v << "\n";
		}
		o << "conn_map: \n";
		for (auto &v : conn_map) {
			o << "\t" <<  v.first.first << "->" << v.first.second << " " << v.second << "\n";
		}
		o << "dt: " << dt << "\n";
		o << "seed: " << seed << "\n";
		o << "neurons_to_listen: ";
		for (auto &v : neurons_to_listen) { o << v << ", "; }
		o << "files: \n";
		for (auto &v : files) {
			o << "\t" <<  v.first << "->" << v.second << "\n";
		}
		o << "reward_dynamics: " << reward_dynamics << "\n";
		o << "\n";
	}

	static string getDeaultConnectionConf() {
		Value cv(kObjectType);
		cv.AddMember("type", "", Json::d.GetAllocator());
		cv.AddMember("dendrite_delay", 0, Json::d.GetAllocator());
		cv.AddMember("start_weight", 0.0, Json::d.GetAllocator());
		cv.AddMember("synapse", "", Json::d.GetAllocator());
		cv.AddMember("inh_synapse", "", Json::d.GetAllocator());
		return Json::stringify(cv);
	}

	static string getDeaultLayerConf() {
		Value cv(kObjectType);
		cv.AddMember("size", 0, Json::d.GetAllocator());
		cv.AddMember("axon_delay", 0, Json::d.GetAllocator());
		cv.AddMember("input", "", Json::d.GetAllocator());
		cv.AddMember("neuron", "", Json::d.GetAllocator());
		cv.AddMember("act_function", "", Json::d.GetAllocator());
		cv.AddMember("learning_rule", "", Json::d.GetAllocator());
		cv.AddMember("weight_normalization", "", Json::d.GetAllocator());
		cv.AddMember("reinforcement", "", Json::d.GetAllocator());
		return Json::stringify(cv);
	}
};


struct Constants : public Printable {
	enum ReadMod {FromString, FromFile};

	Constants(OptMods mods = OptMods());
	Constants(const vector<string> &files, OptMods mods = OptMods()) : Constants(mods) {
		Document d;
		for(const auto& f: files) {
	        std::ifstream ifs(f);
	        std::string const_json(
	        	(std::istreambuf_iterator<char>(ifs)),
                std::istreambuf_iterator<char>()
            );
            // cout << *this;
            readString(const_json, mods);
		}
	}

	void readJson(Document &document);
	void readString(string s, OptMods mods = OptMods());

	static void fill(const Value &v, map<string, string> &m, const string part_name) {
		if(!v.IsObject()) {
			throw dnnException() << "Got strange value for \"" << part_name << "\" part of constants\n";
		}
		for (Value::ConstMemberIterator itr = v.MemberBegin(); itr != v.MemberEnd(); ++itr) {
			m[itr->name.GetString()] = Json::stringify(itr->value);
		}
	}

	void print(ostream &o) const {
		print_section("inputs: ", inputs, o);
		print_section("neurons: ", neurons, o);
		print_section("act_functions: ", act_functions, o);
		print_section("synapses: ", synapses, o);
		print_section("learning_rules: ", learning_rules, o);
		print_section("weight_normalizations: ", weight_normalizations, o);
		print_section("connections: ", connections, o);
		print_section("reinforcements: ", reinforcements, o);
		o << "sim_configuration: \n";
		o << sim_conf;
	}

	static void print_section(const string &sect_name, const map<string, string> &m, ostream &o) {
		o << sect_name << "\n";
		for (auto it = m.begin(); it != m.end(); ++it) {
			o << "\t" << it->first << " :";
			auto spl_doc = split(it->second, '\n');
			if(spl_doc.size()>0) {
				o << " " << spl_doc[0] << "\n";
			}
			for(size_t i=1; i<spl_doc.size(); ++i) {
				o << "\t" <<  spl_doc[i] << "\n";
			}
		}
	}


	map<string, string> neurons;
	map<string, string> act_functions;
	map<string, string> synapses;
	map<string, string> inputs;
	map<string, string> learning_rules;
	map<string, string> weight_normalizations;
	map<string, string> connections;
	map<string, string> reinforcements;
	SimConfiguration sim_conf;
};

}
