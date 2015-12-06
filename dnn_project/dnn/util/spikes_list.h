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
	SpikesList() {
		mutName() = "SpikesList";
	}

	SpikesList(const size_t& size) {
		mutName() = "SpikesList";
		seq.resize(size);
	}


	SpikesListInfo getInfo() {
		SpikesListInfo info;
		info.size = seq.size();
		return info;
	}

	void serial_process() {
		begin() << "info: " << info;

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

	void changeTimeDelta(const double &dt) {
		info.changeTimeDelta(dt);
		for(auto &s: seq) {
			for(auto &t: s.values) {
				t = ceil(t/dt);
			}
		}
	}

	const double& getTimeDelta() const {
		return info.dt;
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

	template <typename CP = FactoryCreationPolicy>
	vector<Ptr<SpikesList>> chop()  {
	    vector<Ptr<SpikesList>> sl_chopped;

	    vector<size_t> indices(size(), 0);

	    for(size_t li=0; li<info.labels_start.size(); ++li) {
            const size_t &start_of_label = info.labels_start[li].second;
            const size_t &label_id = info.labels_start[li].first;

            const size_t &end_of_label = start_of_label + info.unique_labels[label_id].second; // duration
            const string &label = info.unique_labels[label_id].first;


	        Ptr<SpikesList> labeled_sl(CP::template create<SpikesList>());
	        for(size_t di=0; di<size(); ++di) {
	        	while( (indices[di] < seq[di].values.size()) && (seq[di].values[indices[di]] < end_of_label) ) {
	                if(seq[di].values[indices[di]] >= start_of_label) {
		                labeled_sl->addSpike(di, seq[di].values[indices[di]] - static_cast<double>(start_of_label));
	                }
	                indices[di]++;
	        	}
	        }
	        while(size() > labeled_sl->size()) {
	        	labeled_sl->seq.emplace_back();
	        }
	        labeled_sl->info.addLabelAtPos(label, 0, info.unique_labels[label_id].second);
	        sl_chopped.push_back(labeled_sl);
	    }
	    L_DEBUG << "SpikesList, Successfully chopped spike lists in " << sl_chopped.size() << " chunks";
	    return sl_chopped;
	}

	TimeSeriesInfo info;
	vector<SpikesSequence> seq;
};




}