#pragma once

#include <vector>

using std::vector;

#include <dnn/base/base.h>
#include <sys/stat.h>
#include <dnn/contrib/rapidjson/document.h>



namespace dnn {

using namespace rapidjson;

vector<string> split_into(const string &s, char delim, vector<string> &elems);
vector<string> split(const string &s, char delim);
vector<string> splitBySubstr(const string &s_inp, const string &delimiter, const string &not_include = "");

void trim(string &str, string symbols = " \t");
string trimC(const string &str, string symbols = " \t");

#ifndef PI
    #define PI 3.1415926535897932384626433832795028841971693993751
#endif

double getUnif();
double getUnifBetween(double low, double high);
double getExp(double rate);

double sampleDelay(double gain, double rate);
double getNorm();
string strip_white(const string& input);
string strip_comments(const string& input, const string& delimiters);
long getFileSize(string filename);
bool fileExists(const std::string& name);
bool strStartsWith(const string &s, const string &prefix);
bool strEndsWith(const std::string &str, const std::string &suffix);

struct IndexSlice {
    IndexSlice(size_t _from, size_t _to) : from(_from), to(_to) {}
    size_t from;
    size_t to;
};
void replaceStr( string &s, const string &search, const string &replace, const size_t num = std::numeric_limits<size_t>::max());

vector<IndexSlice> dispatchOnThreads(size_t elements_size, size_t jobs);

#define TRY(X) \
	try {	\
		X;	\
	} catch {	\
		throw dnnException()<< "Error!\n"; \
	}\


vector<double> parseParenthesis(const string &s);
map<string, string> parseArgOptionsPairs(const vector<string> &opts);

unsigned long upper_power_of_two(unsigned long v);


ostream& printNow(ostream &o);

template <typename Return, typename ...Parameters, typename ...Args>
auto AssertiveCall(Return (*function)(Parameters...), Args&& ...args)
    -> Return
{
    if (function != nullptr) {
        return (*function)(std::forward<Args>(args)...);
    } else {
        return Return{};
    }
}
void printBackTrace();

double atomicDoubleAdd(std::atomic<double> &f, double d);

}