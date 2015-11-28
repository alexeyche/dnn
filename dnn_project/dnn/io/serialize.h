#pragma once

#include <dnn/util/util.h>
#include <dnn/util/json.h>
#include <dnn/util/interfaced_ptr.h>
#include <dnn/util/ptr.h>
#include <dnn/util/act_vector.h>
#include <dnn/base/base.h>
#include <dnn/protos/base.pb.h>


#include <google/protobuf/message.h>

typedef google::protobuf::Message* ProtoMessage;

namespace dnn {

class Factory;

class SerializableBase  : public Object {
friend class Factory;
public:
    enum ProcessMode { ProcessingInput, ProcessingOutput };
    enum EndMarker { end };

    typedef SerializableBase Self;

    static const bool hasProto = false;
    typedef Protos::EmptyProto ProtoType;

    SerializableBase() : mode(ProcessingOutput), messages(nullptr), header(nullptr) {

    }

    virtual ProtoMessage newProto() {
        throw dnnException()<< "That shouldn't be called. This method for non protobuf Serializable classes\n";
    }

    virtual ~SerializableBase() {
        if(mode == ProcessingOutput) {

        }
    }

    void clean() {
        if(messages) {
            for(auto &m: *messages) {
                delete m;
            }
            delete messages;
            messages = nullptr;
        }
    }

    virtual void serial_process() = 0;
    
    const string& name() const {
        return _name;
    }

    static ProtoMessage copyM(ProtoMessage m) {
        ProtoMessage copy_m = m->New();
        copy_m->CopyFrom(*m);
        return copy_m;
    }
    static Protos::ClassName* getHeader(vector<ProtoMessage> &messages) {
        if(messages.size() == 0) {
            throw dnnException()<< "Trying to get header from empty messages stack\n";
        }
        Protos::ClassName *head = dynamic_cast<Protos::ClassName*>(messages.back());
        if(!head) {
            throw dnnException()<< "There is no header on the top of the stack\n" \
                                << "Got " << messages.back()->GetTypeName() << "\n";
        }
        return head;
    }

    Protos::ClassName* getHeader() {
        if(messages) {
            return getHeader(*messages);
        }
        throw dnnException()<< "Null messages stack\n";
    }

    SerializableBase& begin() {
        assert(messages);
        if(mode == ProcessingOutput) {
            header = new Protos::ClassName;

            header->set_class_name(name());
            header->set_has_proto(false);
            header->set_size(0);

            messages->push_back(header);
        }
        if(mode == ProcessingInput) {
            Protos::ClassName *head = getHeader();
            if(name() != head->class_name()) {
                throw dnnException()<< "Error while deserializing. Wrong class name header: " << name() << " != " << head->class_name() << "\n";
            }
            deleteCurrentMessage();
        }
        return *this;
    }
    void operator << (EndMarker e) {
        if(mode == ProcessingInput) {
            //deleteCurrentMessage();
        }
    }



    SerializableBase& operator << (SerializableBase &b) {
        if(mode == ProcessingOutput) {
            header->set_size(header->size()+1);
            for(auto &m: b.getSerialized()) {
                addMessage(m);
            }
        } else
        if(mode == ProcessingInput) {
            b.getDeserialized(*messages);
        }

        return *this;
    }

    static Ptr<SerializableBase> createObject(string name);

    template <typename T>
    SerializableBase& operator << (InterfacedPtr<T> &b) {
        if(mode == ProcessingOutput) {
            if(!b.isSet()) {
                throw dnnException()<< "Failed to serialize InterfacePtr: it is without an pointer\n";
            }
            (*this) << b.ref();
        } else
        if(mode == ProcessingInput) {
            Ptr<SerializableBase> pb = SerializableBase::createObject(getHeader()->class_name());

            T* p = dynamic_cast<T*>(pb.ptr());
            if(!p) {
                throw dnnException()<< name() << ": cast error while deserializing interfaced ptr, got " << pb->name() << "\n";
            }
            b.set(p);
            (*this) << b.ref();
        }
        return *this;
    }

    vector<ProtoMessage>& getSerialized() {
        if ((mode == ProcessingOutput) && (messages)) clean();
        mode = ProcessingOutput;

        messages = new vector<ProtoMessage>;


        serial_process();
        return *messages;
    }

    void getDeserialized(vector<ProtoMessage> &inp_mess) {
        mode = ProcessingInput;

        if(messages) clean();
        messages = &inp_mess;

        serial_process();
    }

    SerializableBase& operator << (const char *vraw) {
        return *this;
    }
    void addMessage(ProtoMessage m) {
        assert(messages);
        messages->push_back( copyM(m) );
    }

    ProtoMessage currentMessage() {
        assert(messages);
        if(messages->size() == 0) {
            throw dnnException()<< "Trying to get from empty vector of messages\n";
        }
        return messages->back();
    }
    void deleteCurrentMessage() {
        assert(messages);
        if(!messages->empty()) {
            // cout << name() << " stack: \n\t";
            // for(size_t i=0; i<(messages->size()-1); ++i) {
            //     cout << (*messages)[i]->GetTypeName() << ", ";
            // }
            //cout << " || " << messages->back()->GetTypeName() << "\n";

            delete messages->back();
            messages->pop_back();
        }
    }
    virtual void setAsInput(Ptr<SerializableBase> b) {

    }

    // SerializableBase(const SerializableBase &obj) {
    // SerializableBase(const SerializableBase &obj) {

    // }
    // SerializableBase& operator =(const SerializableBase &obj) { return *this; }


    static string& mutName() {
        return _name;
    }
protected:

    vector<ProtoMessage> *messages;
    Protos::ClassName *header;
    ProcessMode mode;

    static string _name;
};

void protobinSave(SerializableBase *b, const string fname);

template <typename Proto>
class Serializable : public SerializableBase {
public:
    #define ASSERT_FIELDS() \
    if((messages->size() == 0)||(!field_descr)) {\
        throw dnnException()<< "Wrong using of Serializable class.\n"; \
    }\

    typedef Serializable<Proto> Self;
    typedef Proto ProtoType;
    static const bool hasProto = true;

    const string name() const {
        Proto _fake_m;
        vector<string> spl = split(_fake_m.GetTypeName(), '.');
        if(spl[0] != "Protos") {
            throw dnnException()<< "Expection Protos:: typename\n";
        }
        string ret;
        for(size_t i=1; i<spl.size(); ++i) {
            ret += spl[i];
        }
        return ret;
    }

    ProtoMessage newProto() {
        return new Proto;
    }

    Serializable& operator << (const char *vraw) {
        if(messages->size() == 0) {
            throw dnnException()<< "Serialaling without begin()\n";
        }

        string v = string(vraw);
        if(trimC(v) == ",") return *this;

        vector<string> v_spl = split(v, ':');
        string fname = v_spl[0];
        trim(fname);

        //cout << messages->size() << "\n";
        // cout << "Filling fname " << fname << " (" << currentMessage()->GetTypeName()  << ")\n";
        const google::protobuf::Descriptor* descriptor = currentMessage()->GetDescriptor();
        field_descr = descriptor->FindFieldByName(fname);

        if(!field_descr) {
            throw dnnException()<< "Can't find proto field by name \"" << fname << "\"\n";
        }

        return *this;
    }
    #define SERIALIZE_METHOD(type, pbmethod_set, pbmethod_get) \
        Serializable& operator << (type &v) { \
            ASSERT_FIELDS() \
            if(mode == ProcessingOutput) { \
                currentMessage()->GetReflection()->pbmethod_set(currentMessage(), field_descr, v); \
            } else { \
                if(field_descr->label() == google::protobuf::FieldDescriptor::LABEL_REPEATED) { \
                    throw dnnException() << "Using method for serializing simple field for repeated one\n"; \
                } \
                if(currentMessage()->GetReflection()->HasField(*currentMessage(), field_descr)) { \
                    v = currentMessage()->GetReflection()->pbmethod_get(*currentMessage(), field_descr); \
                } \
            } \
            return *this; \
        }

    #define SERIALIZE_REPEATED_METHOD(vtype, type, pbmethod_add, pbmethod_get) \
        Serializable& operator << (vtype<type> &v) { \
            ASSERT_FIELDS() \
            if(mode == ProcessingOutput) { \
                for(size_t i=0; i<v.size(); ++i) { \
                    currentMessage()->GetReflection()->pbmethod_add(currentMessage(), field_descr, v[i]); \
                } \
            } else { \
                int cur = currentMessage()->GetReflection()->FieldSize(*currentMessage(), field_descr); \
                for(int i=0; i<cur; ++i) { \
                    type subv = currentMessage()->GetReflection()->pbmethod_get(*currentMessage(), field_descr, i); \
                    v.push_back(subv); \
                } \
            } \
            return *this; \
        }

    SERIALIZE_METHOD(double, SetDouble, GetDouble);
    SERIALIZE_METHOD(bool, SetBool, GetBool);
    SERIALIZE_METHOD(size_t, SetUInt32, GetUInt32);
    SERIALIZE_METHOD(int, SetInt32, GetInt32);
    SERIALIZE_METHOD(string, SetString, GetString);
    SERIALIZE_REPEATED_METHOD(vector, string, AddString, GetRepeatedString);
    SERIALIZE_REPEATED_METHOD(vector, size_t, AddUInt32, GetRepeatedUInt32);
    SERIALIZE_REPEATED_METHOD(vector, double, AddDouble, GetRepeatedDouble);
    SERIALIZE_REPEATED_METHOD(ActVector, size_t, AddUInt32, GetRepeatedUInt32);
    SERIALIZE_REPEATED_METHOD(ActVector, double, AddDouble, GetRepeatedDouble);

    Serializable& operator << (vector<std::complex<double>> &v) {
        ASSERT_FIELDS()
        const google::protobuf::Descriptor* descriptor = currentMessage()->GetDescriptor();
        const google::protobuf::FieldDescriptor* imag_field_descr = descriptor->FindFieldByName(field_descr->name() + "_imag");
        if(!imag_field_descr) {
            throw dnnException() << "Failed to find protobuf field for imaginery part of complex number\n";
        }
        if(mode == ProcessingOutput) {
            for(size_t i=0; i<v.size(); ++i) {
                currentMessage()->GetReflection()->AddDouble(currentMessage(), field_descr, v[i].real());
                currentMessage()->GetReflection()->AddDouble(currentMessage(), imag_field_descr, v[i].imag());
            }
        } else {
            int cur = currentMessage()->GetReflection()->FieldSize(*currentMessage(), field_descr);
            for(int i=0; i<cur; ++i) {
                double subv = currentMessage()->GetReflection()->GetRepeatedDouble(*currentMessage(), field_descr, i);
                double subv_imag = currentMessage()->GetReflection()->GetRepeatedDouble(*currentMessage(), imag_field_descr, i);
                v.push_back(std::complex<double>(subv, subv_imag));
            }
        }
        return *this;
    }



    void operator << (EndMarker e) {
        if(mode == ProcessingInput) {
            deleteCurrentMessage();
        }
    }


    Serializable& begin() {
        if(mode == ProcessingOutput) {
            header = new Protos::ClassName;

            header->set_class_name(name());
            header->set_has_proto(true);
            header->set_size(0);

            messages->push_back(header);

            ProtoMessage mess = new Proto;

            messages->push_back(mess);
        }
        if(mode == ProcessingInput) {
            Protos::ClassName *head = getHeader();
            if(name() != head->class_name()) {
                throw dnnException()<< "Error while deserializing. Wrong class name header: " << name() << " != " << head->class_name() << "\n";
            }
            deleteCurrentMessage();
        }
        return *this;
    }


private:
    const google::protobuf::FieldDescriptor* field_descr;
};

template <typename T>
T* as(SerializableBase *b) {
    T* p = dynamic_cast<T*>(b);
    if(!p) {
        throw dnnException() << "Failed to cast: " << b->name() << "\n";
    }
    return p;
}




}
