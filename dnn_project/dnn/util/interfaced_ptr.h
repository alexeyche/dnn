#pragma once

#include <dnn/util/ptr.h>

namespace dnn {




template <typename T>
class InterfacedPtr {
public:
	InterfacedPtr(T *ptr_to_set) : _ptr(ptr_to_set) {
		_ptr->template provideInterface<T>(i);
	}

	InterfacedPtr() : _ptr(nullptr) {
		T::provideDefaultInterface(i);
	}

	inline T* ptr() {
		return _ptr;
	}
	inline T& ref() {
		return *_ptr;
	}
	inline const T* ptr() const {
		return _ptr;
	}
	inline const T& ref() const {
		return *_ptr;
	}
	void set(T *ptr_to_set) {
		_ptr = ptr_to_set;
		_ptr->template provideInterface<T>(i);
	}

	InterfacedPtr<T>& operator = (Ptr<T> src) {
		set(src.ptr());
		return *this;
	}

	bool isSet() {
		return _ptr ? true : false;
	}

	const typename T::interface& ifc() const {
		return i;
	}

private:
	T *_ptr;
	typename T::interface i;
};




}
