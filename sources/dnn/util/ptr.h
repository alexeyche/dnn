#pragma once

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
private:
	T *_ptr;
};

}