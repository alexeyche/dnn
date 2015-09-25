
#include "constants.h"


const char default_const_json[] = {
    #include "const.xxd"
};



namespace dnn {

Constants::Constants(OptMods mods) : Constants(string(default_const_json), mods, Constants::FromString) {
    sim_conf = SimConfiguration();
}

Constants::Constants(string s, OptMods mods, ReadMod mod) {
    string const_json;
    if(mod == FromFile) {
        std::ifstream ifs(s);
        const_json = std::string((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());
    } else
    if(mod == FromString) {
        const_json = s;
    }
    for(auto it=mods.begin(); it != mods.end(); ++it) {
        replaceStr(const_json, it->first, it->second);
    }

    Document document = Json::parseString(const_json);

    fill(Json::getVal(document, "neurons"), neurons);
    fill(Json::getVal(document, "act_functions"), act_functions);
    fill(Json::getVal(document, "synapses"), synapses);
    fill(Json::getVal(document, "inputs"), inputs);
    fill(Json::getVal(document, "learning_rules"), learning_rules);
    fill(Json::getVal(document, "weight_normalizations"), weight_normalizations);
    fill(Json::getVal(document, "connections"), connections);

    const Value &sim_conf_doc = Json::getVal(document, "sim_configuration");
    const Value &layers_doc = Json::getArray(sim_conf_doc, "layers");

    for (SizeType i = 0; i < layers_doc.Size(); i++) {
        const Value &v = layers_doc[i];
        sim_conf.layers.push_back(Json::stringify(v));
    }
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
    sim_conf.dt = Json::getDoubleVal(sim_conf_doc, "dt");
    sim_conf.seed = Json::getIntVal(sim_conf_doc, "seed");
    if(sim_conf.seed < 0) {
        std::srand ( unsigned ( std::time(0) ) );
    } else {
        std::srand ( sim_conf.seed );
    }
    sim_conf.neurons_to_listen = Json::getUintVector(sim_conf_doc, "neurons_to_listen");
    const Value &files_doc = Json::getVal(sim_conf_doc, "files");
    for (Value::ConstMemberIterator itr = files_doc.MemberBegin(); itr != files_doc.MemberEnd(); ++itr) {
        const string k = itr->name.GetString();
        sim_conf.files[k] = Json::stringify(itr->value);
    }
}


}