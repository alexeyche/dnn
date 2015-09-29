
#include "constants.h"


const char default_const_json[] = {
    #include "const.xxd"
};



namespace dnn {

Constants::Constants(OptMods mods) {
    readString(default_const_json, mods);
    sim_conf = SimConfiguration();
}

void Constants::readString(string s, OptMods mods) {
    for(auto it=mods.begin(); it != mods.end(); ++it) {
        replaceStr(s, it->first, it->second);
    }
    Document document = Json::parseString(s);
    readJson(document);
}

void Constants::readJson(Document &document) {
    #define SAFE_FILL(part) \
        if(Json::checkVal(document, #part)) { \
            fill(Json::getVal(document, #part), part); \
        } \

    SAFE_FILL(neurons);
    SAFE_FILL(act_functions);
    SAFE_FILL(synapses);
    SAFE_FILL(inputs);
    SAFE_FILL(learning_rules);
    SAFE_FILL(weight_normalizations);
    SAFE_FILL(connections);

    if(Json::checkVal(document, "sim_configuration")) {
        const Value &sim_conf_doc = Json::getVal(document, "sim_configuration");
        if(Json::checkVal(sim_conf_doc, "layers")) {
            const Value &layers_doc = Json::getArray(sim_conf_doc, "layers");

            for (SizeType i = 0; i < layers_doc.Size(); i++) {
                const Value &v = layers_doc[i];
                sim_conf.layers.push_back(Json::stringify(v));
            }
        }
        if(Json::checkVal(sim_conf_doc, "conn_map")) {
            const Value &conn_map_doc = Json::getVal(sim_conf_doc, "conn_map");

            for (Value::ConstMemberIterator itr = conn_map_doc.MemberBegin(); itr != conn_map_doc.MemberEnd(); ++itr) {
                const string k = itr->name.GetString();
                vector<string> aff = splitBySubstr(k, "->");
                if (aff.size() != 2) {
                    throw dnnException() << "conn_map configuration not right: need 2 afferents separated by \"->\"\n";
                }

                const pair<size_t, size_t> aff_p(stoi(aff[0]), stoi(aff[1]));
                const Value &conns = itr->value;

                for (SizeType i = 0; i < conns.Size(); i++) {
                    const Value &v = conns[i];
                    sim_conf.conn_map.insert( pair<pair<size_t, size_t>, string>(aff_p, Json::stringify(v) ));
                }
            }
        }
        if(Json::checkVal(sim_conf_doc, "dt")) {
            sim_conf.dt = Json::getDoubleVal(sim_conf_doc, "dt");

        }
        if(Json::checkVal(sim_conf_doc, "seed")) {
            sim_conf.seed = Json::getIntVal(sim_conf_doc, "seed");
            if(sim_conf.seed < 0) {
                std::srand ( unsigned ( std::time(0) ) );
            } else {
                std::srand ( sim_conf.seed );
            }
        }
        if(Json::checkVal(sim_conf_doc, "neurons_to_listen")) {
            sim_conf.neurons_to_listen = Json::getUintVector(sim_conf_doc, "neurons_to_listen");
        }
        if(Json::checkVal(sim_conf_doc, "files")) {
            const Value &files_doc = Json::getVal(sim_conf_doc, "files");
            for (Value::ConstMemberIterator itr = files_doc.MemberBegin(); itr != files_doc.MemberEnd(); ++itr) {
                const string k = itr->name.GetString();
                sim_conf.files[k] = Json::stringify(itr->value);
            }
        }
    }

}

}