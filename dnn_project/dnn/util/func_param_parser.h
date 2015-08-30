#pragma once

#include <map>
#include <string>
#include <functional>
#include <vector>

using std::map;
using std::string;
using std::function;
using std::vector;

#include <dnn/util/util.h>

namespace dnn {



class FuncParamParser {
public:
    class TParserMap {
        typedef vector<pair<string, function<void(string)>>> TVec;
    public:
        function<void(string)>& operator[](string s) {
            acc.resize(acc.size()+1);
            acc.back().first = s;
            return acc.back().second;
        }
        TVec::const_iterator begin() const {
            return acc.cbegin();
        }
        TVec::const_iterator end() const {
            return acc.cend();
        }
    private:
        TVec acc;
    };
    static void parse(string spec, const TParserMap &callbacks);

    static std::function<void(string)> genDoubleParser(double &v) {
        return [&] (string s) { v = std::stof(s); };
    }
    static std::function<void(string)> genStringParser(string &v) {
        return [&] (string s) { v = s; };
    }
};




}