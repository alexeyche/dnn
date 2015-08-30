#pragma once


#include "time_series.h"

#include <dnn/util/ptr.h>
#include <dnn/protos/spikes_list.pb.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct SpikesSequence : public Serializable<Protos::SpikesSequence> {
	void serial_process() {
		begin() << "values: " << values << Self::end;
	}

	vector<double> values;
};

/*@GENERATE_PROTO@*/
struct SpikesListInfo : public Serializable<Protos::SpikesListInfo> {
	void serial_process() {
		begin() << "size: " << size << Self::end;
	}

	size_t size;
};


struct SpikesList : public SerializableBase {
	SpikesList() {}

	SpikesList(const size_t& size) {
		seq.resize(size);
	}

	const string name() const {
		return "SpikesList";
	}

	SpikesListInfo getInfo() {
		SpikesListInfo info;
		info.size = seq.size();
		return info;
	}

	void serial_process() {
		begin() << "ts_info: " << ts_info;

		SpikesListInfo info;
		if (mode == ProcessingOutput) {
			info = getInfo();
		}

		(*this) << "SpikesList: "  << info;

		if (mode == ProcessingInput) {
			seq.resize(info.size);
		}

		for(size_t i=0; i<info.size; ++i) {
			(*this) << seq[i];
		}
		(*this) << Self::end;
	}

	vector<double>& operator [](const size_t &i) {
		return seq[i].values;
	}

	inline const size_t size() const {
		return seq.size();
	}
	Ptr<TimeSeries> convertToBinaryTimeSeries(const double &dt) const;

	void addSpike(size_t ni, double t) {
		while(ni >= seq.size()) {
			seq.emplace_back();
		}
		if((seq[ni].values.size()>0)&&(seq[ni].values.back()>t)) {
			throw dnnException() << "Adding spike in past to spikes list. Add sorted array of spikes\n";
		}
		seq[ni].values.push_back(t);
	}


	TimeSeriesInfo ts_info;
	vector<SpikesSequence> seq;
};




}