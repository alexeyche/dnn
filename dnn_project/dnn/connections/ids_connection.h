#pragma once

#include <set>
#include <map>

#include <dnn/connections/connection.h>
#include <dnn/protos/ids_connection.pb.h>

using std::map;
using std::set;


namespace dnn {

/*@GENERATE_PROTO@*/
struct IdsConnectionC : public Serializable<Protos::IdsConnectionC> {
    IdsConnectionC() {}

    void serial_process() {
        begin() << "ids: " << ids << Self::end;
        if(mode == ProcessingInput) {
            for(auto id_spec: ids) {
                pair<size_t, size_t> aff = parseConnectionSpec(id_spec);
                auto left_id = __ids.find(aff.first);
                if(left_id == __ids.end()) {
                    __ids[aff.first] = set<size_t>();
                }
                __ids[aff.first].insert(aff.second);
            }
        }
    }

    vector<string> ids;

    map<size_t, set<size_t>> __ids;
};


class IdsConnection : public Connection<IdsConnectionC> {
public:
    const string name() const {
        return "IdsConnection";
    }

    ConnectionRecipe getConnectionRecipe(const SpikeNeuronBase &left, const SpikeNeuronBase &right) {
        ConnectionRecipe recipe;
        auto left_id = c.__ids.find(left.localId());
        if(left_id != c.__ids.end()) {
            auto right_id = left_id->second.find(right.localId());
            if(right_id != left_id->second.end()) {
                L_DEBUG << name() << ", " << "Connecting neurons " << left.localId() << " with " << right.localId();
                recipe.exists = true;
            }
        }
        return recipe;
    }
};





};