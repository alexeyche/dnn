#pragma once


#include <iostream>
#include <memory>
#include <numeric>
#include <functional>
#include <sstream>
#include <string>
#include <map>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <google/protobuf/message.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdlib>
#include <queue>
#include <atomic>
#include <stdexcept>
#include <exception>
#include <unordered_set>
#include <set>
#include <future>
#include <iterator>
#include <complex>


typedef google::protobuf::Message* ProtoMessagePtr;

using std::map;
using std::set;
using std::cout;
using std::move;
using std::accumulate;
using std::stringstream;
using std::istringstream;
using std::ostringstream;
using std::string;
using std::ostream;
using std::istream;
using std::cin;
using std::pair;
using std::min;
using std::ofstream;
using std::ifstream;
using std::multimap;
using std::priority_queue;
using std::unordered_set;
using std::complex;

template <typename T>
using uptr = std::unique_ptr<T>;

template <typename T>
using sptr = std::shared_ptr<T>;

