
#include <dnn/base/base.h>
#include <dnn/neurons/leaky_integrate_and_fire.h>
#include <dnn/neurons/adapt_integrate_and_fire.h>
#include <dnn/neurons/spike_sequence_neuron.h>
#include <dnn/neurons/srm_neuron.h>
#include <dnn/act_functions/determ.h>
#include <dnn/act_functions/exp_threshold.h>
#include <dnn/synapses/static_synapse.h>
#include <dnn/synapses/std_synapse.h>
#include <dnn/inputs/input_time_series.h>
#include <dnn/inputs/white_noise_input.h>
#include <dnn/io/serialize.h>
#include <dnn/util/time_series.h>
#include <dnn/util/statistics.h>
#include <dnn/util/matrix.h>
#include <dnn/util/spikes_list.h>
#include <dnn/learning_rules/stdp.h>
#include <dnn/learning_rules/optimal_stdp.h>
#include <dnn/learning_rules/stdp_time.h>
#include <dnn/learning_rules/triple_stdp.h>
#include <dnn/weight_normalizations/pow_min_max.h>
#include <dnn/weight_normalizations/min_max.h>
#include <dnn/weight_normalizations/nonlinear_min_max.h>
#include <dnn/weight_normalizations/sliding_ltd.h>
#include <dnn/reinforcements/input_classifier.h>
#include <dnn/sim/reward_control.h>
#include <dnn/connections/stochastic.h>
#include <dnn/connections/difference_of_gaussians.h>
#include <dnn/connections/ids_connection.h>


#include <dnn/util/log/log.h>


#include <dnn/sim/sim.h>



#include "factory.h"

namespace dnn {

Factory::entity_map_type Factory::typemap;
Factory::proto_map_type Factory::prototypemap;

Factory& Factory::inst() {
	static Factory _inst;
	return _inst;
}


class BasicTypeDeduce : public TypeDeducer {
public:
    string deduceType(const std::type_info &info) const {
        #define REG_FILE <dnn/base/register.x>
        #include <dnn/base/deduce_type_impl.x>
        #undef REG_FILE
    }
};


Factory::Factory() {
    Log::inst().setLogLevel(Log::DEBUG_LEVEL);

    #define REG_FILE <dnn/base/register.x>
    #include <dnn/base/register_impl.x>
    #undef REG_FILE

    addTypeDeducer(new BasicTypeDeduce());
}




Factory::~Factory() {
    cleanHeap();
    for(auto &td: type_deducers) {
        delete td;
    }
}


// SerializableBase* Factory::createObject(string name) {
// 	if (typemap.find(name) == typemap.end()) {
// 		throw dnnException()<< "Failed to find method to construct type " << name << "\n";
// 	}
// 	SerializableBase* o = typemap[name]();

// 	if(registration_is_on) {
// 		objects.push_back(o);
// 		objects_map.insert(std::make_pair(o->name(), objects.size()-1));
// 	}
// 	return o;
// }

ProtoMessage Factory::createProto(string name) {
	if (prototypemap.find(name) == prototypemap.end()) {
		throw dnnException()<< "Failed to find method to construct proto type " << name << "\n";
	}
	ProtoMessage o = prototypemap[name]();
	return o;
}



Ptr<SerializableBase> Factory::getCachedObject(const string& filename) {
    if(cache_map.find(filename) == cache_map.end()) {
        ifstream f(filename);
        Stream s(f, Stream::Binary);

        cache_map[filename] = s.readBase();
    }
    return cache_map[filename];
}


pair<Factory::object_iter, Factory::object_iter> Factory::getObjectsSlice(const string& name) {
	auto res = objects_map.equal_range(name);
    if(res.first == res.second) {
        throw dnnException() << "Can't find object slice with name " << name << " in factory cache\n";
    }
    return res;
}

}
