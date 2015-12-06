#include "stream.h"

#include <dnn/io/serialize.h>
#include <dnn/protos/base.pb.h>
#include <dnn/base/factory.h>

namespace dnn {

// Stream& Stream::operator << (SerializableBase &v) {
//     v.processStream(*this);
//     return *this;
// }




void Stream::writeObject(SerializableBase *b) {
	if (!isOutput()) {
		throw dnnException()<< "Stream isn't open in output mode. Need output stream\n";
	}

	vector<ProtoMessagePtr> messages = b->getSerialized();

	if (getRepr() == Text) {
		vector<string> buff;
		Document d;
		Value o(kObjectType);
		string temps = b->name();
		if(!Factory::inst().isProtoType(temps)) {
			Value sub_o(kObjectType);
			for (auto &m : messages) {
				if(m->GetTypeName() != "Protos.ClassName") {
					Document *sub_d = Json::parseProtobuf(m);
					buff.push_back(m->GetTypeName());
					sub_o.AddMember(StringRef(buff.back().c_str()), *sub_d, d.GetAllocator());
				}
			}
			o.AddMember(StringRef(temps.c_str()), sub_o, d.GetAllocator());
		} else {
			assert(messages.size() == 2);
			Document *sub_d = Json::parseProtobuf(messages[1]);
			o.AddMember(StringRef(temps.c_str()), *sub_d, d.GetAllocator());
		}

		(*_output_str) << Json::stringify(o);
	} else if (getRepr() == Binary) {
		for (auto &m : messages) {
			writeBinaryMessage(m, _output_str);
		}
	}

}

#define P(condition) {if(!condition) { printf( "\n FAILURE in %s, line %d\n", __FILE__, __LINE__ );exit( 1 );}}

void Stream::protoReader(vector<ProtoMessagePtr> &messages) {
	Protos::ClassName *cl = new Protos::ClassName;
	bool ok = readBinaryMessage(cl, _input_str);
	if (!ok) return;

	messages.push_back(cl);

	if (cl->has_proto()) {
		ProtoMessagePtr pr = Factory::inst().createProto(cl->class_name());
		P(readBinaryMessage(pr, _input_str));
		messages.push_back(pr);
	}

	for (size_t i = 0; i < cl->size(); ++i) {
		protoReader(messages);
	}
}

void Stream::jsonReader(string name, const Value &v, vector<ProtoMessagePtr> &messages) {
	// json to protobuf
	vector<string> name_spl = split(name, '_');
	if(name_spl.size()>0) {
		name = name_spl[0];
	}
	Protos::ClassName *cl = new Protos::ClassName;
	cl->set_class_name(name);
	messages.push_back(cl);

	bool has_object = false;
	for (Value::ConstMemberIterator itr = v.MemberBegin(); itr != v.MemberEnd(); ++itr) {
		if (itr->value.IsObject()) {
			jsonReader(itr->name.GetString(), itr->value, messages);
			has_object = true;
		}
	}
	if (!has_object) {
		if (!Factory::inst().isProtoType(name)) {
			// trying to deduce proto name
			if (!Factory::inst().isProtoType(name + string("C"))) {
				throw dnnException()<<  "Erros while reading " << name << ": unkwnown proto type\n";
			}
			name += string("C");
			Protos::ClassName *ccl = new Protos::ClassName;
			ccl->set_class_name(name);
			messages.push_back(ccl);
		}
		ProtoMessagePtr m = Factory::inst().createProto(name);
		Json::JsonToProtobuf(v, m);
		messages.push_back(m);
	}
}

vector<ProtoMessagePtr> Stream::readObjectProtos() {
	vector<ProtoMessagePtr> messages;
	if (r == Binary) {
		protoReader(messages);
	}
	if (r == Text) {
		if (!iterator->value.IsObject()) {
			throw dnnException()<< "Fail to read " << Json::stringify(iterator->value) << ". Expected object\n";
		}
		if (iterator == document.MemberEnd()) {
			return messages;
		}
		jsonReader(iterator->name.GetString(), iterator->value, messages);
		iterator++;
	}
	return messages;
}


Ptr<SerializableBase> Stream::deserialize(string name, vector<ProtoMessagePtr> &messages, bool dynamically, SerializableBase *src) {
    if(messages.size() == 0) {
        return Ptr<SerializableBase>();
    }
    if(!src) {
        Ptr<SerializableBase> new_o;
        if(dynamically) {
            new_o = Factory::inst().createDynamicObject(name); //
        } else {
            new_o = Factory::inst().createObject(name); // head->class_name());
        }
        new_o->getDeserialized(messages);
        return new_o;
    } else {
        src->getDeserialized(messages);
        return Ptr<SerializableBase>(src);
    }
}

}


