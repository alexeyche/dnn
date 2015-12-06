#pragma once

#include <type_traits>
#include <typeinfo>


#include <dnn/core.h>
#include <dnn/io/serialize.h>
#include <dnn/util/ptr.h>

#include <dnn/base/type_deducer.h>

namespace dnn {

class ClassName;
class SerializableBase;


class Factory {
public:
    friend class SerializableBase;
    friend class Stream;

    Factory();


    typedef map<string, SerializableBase* (*)()> entity_map_type;
    typedef map<string, ProtoMessagePtr (*)()> proto_map_type;
    typedef multimap<string, size_t>::iterator object_iter;

    template<typename INST> static SerializableBase* createInstance() { return new INST; }
    template<typename INST> static ProtoMessagePtr createProtoInstance() { return new INST; }

    template<typename T>
    string deduceType() {
        string t;
        for(const auto& deducer: type_deducers) {
            string type = deducer->deduceType(typeid(T));
            if(!type.empty() && !t.empty()) {
                throw dnnException() << "Got ambigous types: " << type << " and " << t << "\n";
            }
            if(!type.empty()) {
                t = type;
            }
        }
        if(t.empty()) {
            throw dnnException() << "Can't deduce type\n";
        }
        return t;
    }

    ~Factory();

    void addTypeDeducer(TypeDeducer *t) {
        type_deducers.push_back(t);
    }

    template <typename T>
    static void registerType(const string type) {
        typemap[type] = &createInstance<T>;
        if (T::hasProto) {
            prototypemap[type] = &createProtoInstance<typename T::ProtoType>;
        }
    }

    bool isProtoType(const string name) {
        return prototypemap.find(name) != prototypemap.end();
    }

    // SerializableBase* createObject(string name);
    ProtoMessagePtr createProto(string name);

    static Factory& inst();

    pair<object_iter, object_iter> getObjectsSlice(const string& name);

    Ptr<SerializableBase> getObject(object_iter &it) {
        return objects[it->second];
    }


    template <typename T>
    Ptr<T> createObject() {
        Ptr<T> o = _createObject<T>();
        registerObject(o.ptr());
        return o;
    }

    template <typename T>
    Ptr<T> createDynamicObject() {
        Ptr<T> o = _createObject<T>();
        o.owner = true;
        return o;
    }

    Ptr<SerializableBase> getCachedObject(const string& filename);
    void registerObject(Ptr<SerializableBase> o) {
        o.owner = false;
        objects.push_back(o);
        objects_map.insert(std::make_pair(o->name(), objects.size()-1));
    }

    void cleanHeap() {
        for (auto &o : objects) {
            o.destroy();
        }
        objects_map.clear();
        cache_map.clear();
    }

private:

    Ptr<SerializableBase> createDynamicObject(string name) {
        if (typemap.find(name) == typemap.end()) {
            throw dnnException()<< "Failed to find method to construct type " << name << "\n";
        }
        SerializableBase* o = typemap[name]();
        o->mutName() = name;
        return Ptr<SerializableBase>(o);
    }


    Ptr<SerializableBase> createObject(string name) {
        Ptr<SerializableBase> o = createDynamicObject(name);
        registerObject(o);
        return o;
    }

    template <typename T>
    Ptr<T> _createObject() {
        string name = deduceType<T>();
        Ptr<SerializableBase> o = createDynamicObject(name);

        T *p = dynamic_cast<T*>(o.ptr());
        if (!p) {
            throw dnnException()<< "Error to cast while creating " << name << " to its base" << "\n";
        }
        return Ptr<T>(p);
    }


    static entity_map_type typemap;
    static proto_map_type prototypemap;

    multimap<string, size_t> objects_map;

    vector<Ptr<SerializableBase>> objects;
    map<string, Ptr<SerializableBase>> cache_map;

    vector<TypeDeducer*> type_deducers;
};

struct DynamicCreationPolicy {
    template <typename T>
    static Ptr<T> create() {
        return Factory::inst().createDynamicObject<T>();
    }
};


struct FactoryCreationPolicy {
    template <typename T>
    static Ptr<T> create() {
        return Factory::inst().createObject<T>();
    }
};



}
