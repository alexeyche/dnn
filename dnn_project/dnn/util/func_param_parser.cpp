#include "func_param_parser.h"

#include <dnn/util/log/log.h>

namespace dnn {


void FuncParamParser::parse(string spec, const TParserMap &callbacks) {
    L_DEBUG << "FuncParamParser, Processing spec: " << spec;

    size_t endpos = spec.find_last_of(")");
    if(string::npos != endpos) {
        spec = spec.substr(0, endpos);
    }
    size_t startpos = spec.find_first_of("(");
    if(string::npos != startpos) {
        spec = spec.substr(startpos+1, string::npos);
    }
    L_DEBUG << "FuncParamParser, Processing cleared spec: " << spec;
    vector<string> split_spec = splitBySubstr(spec, ",", "(");
    //vector<string> split_spec = split(spec, ',');
    auto cb_it = callbacks.begin();
    for(auto s: split_spec) {
        L_DEBUG << "FuncParamParser, Processing part of spec: " << s;
        vector<string> eq_spl = splitBySubstr(s, "=", "(),");
        if(eq_spl.size() == 1) {
            // non named parameter
            if(cb_it == callbacks.end()) {
                throw dnnException() << "Got extra ordered param: " << eq_spl[0] << "\n";
            }
            try {
                L_DEBUG << "FuncParamParser, Calling " << cb_it->first << " with value " << eq_spl[0];
                (*cb_it).second(eq_spl[0]);
            } catch(...) {
                throw dnnException() << "Failed to parse: " << eq_spl[0] << "\n";
            }
            cb_it++;
        } else
        if(eq_spl.size() == 2) {
            string param_name = trimC(eq_spl[0], " \t");

            auto c_param_it = callbacks.begin();
            while(c_param_it != callbacks.end()) {
                if(c_param_it->first == param_name) {
                    break;
                }
                ++c_param_it;
            }
            if(c_param_it == callbacks.end()) {
                throw dnnException() << "Got extra named parameter with name: " << param_name << " and value: " << eq_spl[1] << "\n";
            }
            try {
                L_DEBUG << "FuncParamParser, Calling " << cb_it->first << " with value " << eq_spl[1];
                (*c_param_it).second(eq_spl[1]);
            } catch(...) {
                throw dnnException() << "Failed to parse: " << eq_spl[1] << "\n";
            }

        } else {
            throw dnnException() << "Bad named parameter: " << s << "\n";
        }
    }
}



}