#pragma once

#include <vector>

using std::vector;

#include <dnn/base/base.h>

namespace dnn {

template <typename T>
T cast(string &s) {
	throw dnnException()<< "Can't cast " << s << "\n";
}

template <>
double cast(string &s);

template <>
int cast(string &s);

template <>
size_t cast(string &s);

template <>
string cast(string &s);

template <typename T>
class Accum;

template <>
Accum<string> cast(string &s);


class OptionParser {
public:
	OptionParser(int argc, char **argv) {
		for (size_t i = 1; i < argc; ++i) {
			opts.push_back(argv[i]);
		}
	}
	OptionParser(const vector<string> &args) : opts(args) {}

	template <typename T>
	void option(string long_opt, string short_opt, T &src, bool required = true, bool as_flag = false) {
		bool found = false;
		auto it = opts.begin();
		while (it != opts.end()) {
			if ( (*it == long_opt) || ( (!short_opt.empty()) && (*it == short_opt)) ) {
				if (as_flag) {
					src = true;
					// cout << "Flag " << *it << "\n";
					it = opts.erase(it);

				} else {
					if ( (it + 1) == opts.end() ) {
						throw dnnException()<< "Can't find value for option " << long_opt << "\n";
					}
					// cout << "Opt " << *it << "\n";
					src = cast<T>(*(++it));
					it = opts.erase(it - 1, it + 1);
				}
				found = true;
			} else {
				++it;
			}
		}
		if ((!found) && (required)) {
			throw dnnException()<< "Can't find value for option " << long_opt << "\n";
		}

	}
	template <typename T>
	void loption(string long_opt, T &src, bool required = true, bool as_flag = false) {
		option(long_opt, "", src, required, as_flag);
	}
	vector<string>& getRawOptions() {
		return opts;
	}
	void checkEmpty() {
		if(opts.size() != 0) {
			string s;
			for(auto v: opts) {
				s += " " + v;
			}
			throw dnnException() << "Got unknown options: "  << s << "\n";
		}
	}
private:

	vector<string> opts;
};


}