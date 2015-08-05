#pragma once

#include <vector>

using std::vector;

#include <dnn/core.h>



namespace dnn {


template<typename T>
class ActVector : private vector<T> {
public:
	inline T& operator[] (const size_t &i) {
		makeActive(i);
		return vector<T>::operator[](i);
	}
	void makeActive(const size_t &i) {
		if(act_indices.find(i) == act_indices.end()) {
			act_indices.insert(i);
		}
	}
	inline T& operator[](const unordered_set<size_t>::iterator &i) {
		return vector<T>::operator[](*i);
	}
	inline T& get(const size_t &i) {
		return vector<T>::operator[](i);
	}
	inline const T& get(const size_t &i) const {
		return vector<T>::operator[](i);
	}
	void push_back(const T &v) {
		vector<T>::push_back(v);
	}
	typename vector<T>::iterator begin() {
		return vector<T>::begin();
	}
	typename vector<T>::iterator end() {
		return vector<T>::end();
	}

	typename vector<T>::const_iterator begin() const {
		return vector<T>::cbegin();
	}

	typename vector<T>::const_iterator end() const {
		return vector<T>::cend();
	}

	unordered_set<size_t>::iterator ibegin() {
		return act_indices.begin();
	}
	unordered_set<size_t>::iterator iend() {
		return act_indices.end();
	}
	void setInactive(unordered_set<size_t>::iterator &i) {
		i = act_indices.erase(i);
	}
	size_t size() const {
		return vector<T>::size();
	}
	void resize(size_t s) {
		vector<T>::resize(s);
	}
    friend std::ostream& operator<<(std::ostream& str, const ActVector &self) {
        for(size_t i=0; i<self.size(); ++i) {
        	cout << i << ":";
        	if(self.act_indices.find(i) == self.act_indices.end()) {
        		cout << "n";
        	} else {
        		cout << "a";
        	}
        	cout << ", ";
        }
    	cout << "\n";
        return str;
    }
private:
	unordered_set<size_t> act_indices;
};


}