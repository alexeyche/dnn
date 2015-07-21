#pragma once

#include <dnn/core.h>

namespace dnn {

template <typename T>
class Ptr {
public:
	Ptr(T *ptr_to_set) : _ptr(ptr_to_set) {		
	}

	Ptr() : _ptr(nullptr) {
	}
	inline T* ptr() {
		assert(_ptr);
		return _ptr;
	}
	inline const T* ptr() const {
		assert(_ptr);
		return _ptr;
	}
	inline T& ref() {
		assert(_ptr);
		return *_ptr;
	}
	
	inline const T& ref() const {
		assert(_ptr);
		return *_ptr;
	}
	void set(T *ptr_to_set) {
		_ptr = ptr_to_set;		
	}
	void set(T &ptr_to_set) {
		_ptr = &ptr_to_set;		
	}
	
	const T* operator -> () const {
		return ptr();
	}
	T* operator -> () {
		return ptr();
	}
	const T& operator * () const {
		return ref();
	}
	bool isSet() {
		return _ptr ? true : false;
	}
	explicit operator bool() {
		return isSet();
	}
	
	template<typename NT> 
	Ptr<NT> as() {
		if(!isSet()) return Ptr<NT>();
		NT *t = dynamic_cast<NT*>(_ptr);
		if(!t) return Ptr<NT>();
		return t;
	}
private:
	T *_ptr;
};

}