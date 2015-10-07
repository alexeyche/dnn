#pragma once

#include <dnn/core.h>
#include <dnn/util/log/log.h>
#include <dnn/util/util.h>

#include <typeinfo>


namespace dnn {

class Factory;

template <typename T>
class Ptr {
friend class Factory;
public:
	Ptr(T *ptr_to_set) : _ptr(ptr_to_set), owner(false) {
	}
	~Ptr() {
		destroy();
		// if(owner && _ptr) {
		// 	L_DEBUG << "Found object is not destroyed\n";
		// 	printBackTrace();
		// }
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
	bool isSet() const {
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

	void destroy() {
		if(isSet() && owner) {
			delete _ptr;
			_ptr = nullptr;
		}
	}

private:
	T *_ptr;
	bool owner;
};

}