#pragma once

#include <vector>

using std::vector;

#include <dnn/util/pretty_print.h>
#include <dnn/protos/time_series.pb.h>
#include <dnn/util/ptr.h>
#include <dnn/io/serialize.h>
#include <dnn/base/factory.h>

namespace dnn {







struct TimeSeriesInfo : public Serializable<Protos::TimeSeriesInfo> {
	TimeSeriesInfo(): dt(1.0), __current_position(0) {}
	void serial_process() {
		begin() << "labels_ids: " 	   << labels_ids 	  << ", " \
		        << "unique_labels: "   << unique_labels   << ", " \
		        << "labels_timeline: " << labels_timeline << ", " \
		        << "dt: " << dt << Self::end;

	}

	void addLabelAtPos(const string &lab, size_t pos) {
		size_t lab_id;
		auto ulab_ptr = std::find(unique_labels.begin(), unique_labels.end(), lab);
		if(ulab_ptr == unique_labels.end()) {
			unique_labels.push_back(lab);
			lab_id = unique_labels.size()-1;
		} else {
			lab_id = ulab_ptr - unique_labels.begin();
		}
		labels_ids.push_back(lab_id);
		labels_timeline.push_back(pos);
	}

	void changeTimeDelta(const double &_dt) {
		dt = _dt;
		for(auto &lt: labels_timeline) {
			lt = ceil(lt/dt);
		}
	}

	bool operator == (const TimeSeriesInfo &l) {
		if(labels_ids != labels_ids) return false;
		if(unique_labels != unique_labels) return false;
		if(labels_timeline != labels_timeline) return false;
		return true;
	}
	bool operator != (const TimeSeriesInfo &l) {
		return ! (*this == l);
	}

	const size_t& getClassId(const double &t) {
		while(__current_position < labels_timeline.size()) {
            if(t <= labels_timeline[__current_position]) {
                return labels_ids[__current_position];
            }
            __current_position += 1;
		}
		throw dnnException() << "Trying to get current class for time bigger than Tmax: " << t << "\n";
	}

	vector<size_t> labels_ids;
	vector<string> unique_labels;
	vector<size_t> labels_timeline;
	double dt;

	size_t __current_position;
};


struct TimeSeriesData : public Serializable<Protos::TimeSeriesData> {
	void serial_process() {
		begin() << "values: " << values << Self::end;
	}

	vector<double> values;
};



struct TimeSeriesDimInfo : public Serializable<Protos::TimeSeriesDimInfo> {
	TimeSeriesDimInfo() : size(0) {}
	void serial_process() {
		begin() << "size: " << size << Self::end;
	}

	size_t size;
};

struct TimeSeriesInterface {
	getValueAtIndexDelegate getValueAt;
};

template <typename DATA, typename ELEM>
struct TimeSeriesGeneric {
	TimeSeriesGeneric() {}

	TimeSeriesGeneric(const vector<ELEM> &v) {
		dim_info.size = 1;
		data.resize(dim_info.size);
		data[0].values = v;
	}

	void padRightWithZeros(size_t padSize) {
		for(auto &v: data) {
			for(size_t zi=0; zi<padSize; ++zi) {
				v.values.push_back(ELEM(0.0));
			}
		}
	}

	void cutFromRight(size_t cutSize) {
		for(auto &v: data) {
			for(size_t zi=0; zi<cutSize; ++zi) {
				v.values.pop_back();
			}
		}
	}
	void assertAnotherTs(const TimeSeriesGeneric<DATA, ELEM> &anotherTs) {
		if( (data.size() != anotherTs.data.size()) || (length() != anotherTs.length()) ) {
			throw dnnException() << "Can't multiply time series with different dimenstions\n";
		}
	}

	void norm() {
		for(size_t di=0; di < data.size(); ++di) {
			ELEM acc(0.0);
			for(const auto &v: data[di].values) {
				acc += v*v;
			}
			acc = sqrt(acc);
			for(auto &v: data[di].values) {
				v = v / acc;
			}
		}
	}

	double innerProductMul(TimeSeriesGeneric<DATA, ELEM> &anotherTs) {
		assertAnotherTs(anotherTs);
		double acc = 1.0;
		for(size_t di=0; di < data.size(); ++di) {
			acc *= std::inner_product(data[di].values.begin(), data[di].values.end(), anotherTs.data[di].values.begin(), 0.0);
		}
		return acc;
	}

	double innerProductAcc(TimeSeriesGeneric<DATA, ELEM> &anotherTs) {
		assertAnotherTs(anotherTs);
		double acc = 0.0;
		for(size_t di=0; di < data.size(); ++di) {
			acc += std::inner_product(data[di].values.begin(), data[di].values.end(), anotherTs.data[di].values.begin(), 0.0)/data[di].values.size();
		}
		return acc;
	}

	void operator * (const TimeSeriesGeneric<DATA, ELEM> &anotherTs) {
		if( ((dim() != anotherTs.dim())&&(anotherTs.dim() != 1)) || (length() != anotherTs.length()) ) {
			throw dnnException() << "Can't multiply time series with different dimensions or length\n";
		}
		for(size_t di=0; di < data.size(); ++di) {
			for(size_t vi=0; vi < data[di].values.size(); ++vi) {
				size_t another_ts_di = di;
				if(anotherTs.dim() == 1) {
					another_ts_di = 0;
				}
				data[di].values[vi] *= anotherTs.data[another_ts_di].values[vi];
			}
		}
	}

	vector<ELEM>& getMutVector(size_t ndim) {
		while(ndim >= dim_info.size) {
			dim_info.size = ndim+1;
			data.push_back(DATA());
		}
		assert(dim_info.size == data.size());
		return data[ndim].values;
	}
	const vector<ELEM>& getVector(size_t ndim) const {
		return const_cast<TimeSeriesGeneric<DATA,ELEM>&>(*this).getMutVector(ndim);
	}

	vector<double> getColumnVector(size_t xi) {
		vector<double> col(dim());
		for(size_t di=0; di < dim(); ++di) {
			assert(xi<data[di].values.size());
			col[di] = data[di].values[xi];
		}
		return col;
	}

	void addValue(size_t dim_index, ELEM val) {
		if(dim_index == dim_info.size) {
			dim_info.size = dim_index+1;
			data.push_back(DATA());
			assert(dim_info.size == data.size());
		}
		if(dim_index >= data.size()) {
			throw dnnException() << "Trying to make randow write to memory\n";
		}

		data[dim_index].values.push_back(val);
	}
	size_t length() const {
		if(data.size() == 0) return 0;
		return data[0].values.size();
	}

	size_t dim() const {
		return data.size();
	}

	const ELEM& getValueAt(const size_t &index) const {
		return data[0].values[index];
	}
	const ELEM& getValueAtDim(const size_t &index, const size_t &dim) const {
		return data[dim].values[index];
	}
	const ELEM& operator () (const size_t &dim, const size_t &index) const {
		return data[dim].values[index];
	}

	void assertOneLabel() {
		if(info.unique_labels.size() == 0) {
			throw dnnException() << "Trying to get one label from nonlabeled time series\n";
		}
		if(info.unique_labels.size() > 1) {
			throw dnnException() << "Trying to get one label from multilabeled time series\n";
		}
	}
	const string& getLabel() {
		assertOneLabel();
		return info.unique_labels[0];
	}

	const size_t& getLabelId() {
		assertOneLabel();
		return info.labels_ids[0];
	}
	void setDimSize(const size_t &size) {
		while(dim_info.size < size) {
			data.push_back(DATA());
			dim_info.size = data.size();
		}
	}
	const double& getTimeDelta() const {
		return info.dt;
	}
	void changeTimeDelta(const double &dt) {
		info.changeTimeDelta(dt);
	}


	TimeSeriesDimInfo dim_info;
	TimeSeriesInfo info;
	vector<DATA> data;
};

struct TimeSeries : public SerializableBase, public TimeSeriesGeneric<TimeSeriesData, double> {
	typedef TimeSeriesInterface interface;

	TimeSeries() {}
	TimeSeries(const vector<double> &v) : TimeSeriesGeneric<TimeSeriesData, double>(v) {}
	TimeSeries(const string &filename, const string &format) {
		readFromFile(filename, format);
	}


	void readFromFile(const string &filename, const string &format);

	template <typename T>
	void provideInterface(TimeSeriesInterface &i) {
		i.getValueAt = MakeDelegate(static_cast<T*>(this), &T::getValueAt);
	}
	static const double& getValueAtDefault(const size_t &index) {
		throw dnnException()<< "Calling inapropriate default function method\n";
	}
	static void provideDefaultInterface(TimeSeriesInterface &i) {
		i.getValueAt = &TimeSeries::getValueAtDefault;
	}

	void serial_process() {
		begin() << "dim_info: " << dim_info;
		if (mode == ProcessingInput) {
			data.resize(dim_info.size);
		}
		for(size_t i=0; i<dim_info.size; ++i) {
			(*this) << data[i];
		}
		(*this) << "info: " << info << Self::end;
	}

	static void convertUcrTimeSeriesLine(const string &line, vector<double> &ts_data, string &lab) {
	   vector<string> els = split(line, ' ');
	   assert(els.size() > 0);

	   for(size_t i=0; i<els.size(); i++) {
	       trim(els[i]);
	       if(!els[i].empty()) {
	           if(lab.empty()) {
	               std::ostringstream lab_format;
	               lab_format << stoi(els[i]);
	               lab = lab_format.str();
	               continue;
	           }
	           ts_data.push_back(stof(els[i]));
	       }
	   }
	}

	template <typename CP = FactoryCreationPolicy>
	vector<Ptr<TimeSeries>> chop()  {
	    size_t elem_id = 0;
	    vector<Ptr<TimeSeries>> ts_chopped;
	    assert(info.labels_timeline.size() == info.labels_ids.size());
	    for(size_t li=0; li<info.labels_timeline.size(); ++li) {
	        const size_t &end_of_label = info.labels_timeline[li];
	        const size_t &label_id = info.labels_ids[li];
	        const string &label = info.unique_labels[label_id];

	        Ptr<TimeSeries> labeled_ts(CP::template create<TimeSeries>());
	        for(; elem_id < end_of_label; ++elem_id) {
	            for(size_t di=0; di<data.size(); ++di) {
	                labeled_ts->addValue(di, data[di].values[elem_id]);
	            }
	        }
	        labeled_ts->info.addLabelAtPos(label, labeled_ts->length());
	        ts_chopped.push_back(labeled_ts);
	    }
	    L_DEBUG << "TimeSeries, Successfully chopped time series in " << ts_chopped.size() << " chunks";
	    return ts_chopped;
	}
};


struct TimeSeriesComplexData : public Serializable<Protos::TimeSeriesComplexData> {
	void serial_process() {
		begin() << "values: " << values << Self::end;
	}

	vector<complex<double>> values;
};

struct TimeSeriesComplex : public SerializableBase, public TimeSeriesGeneric<TimeSeriesComplexData, std::complex<double>> {

	void serial_process() {
		begin() << "dim_info: " << dim_info;
		if (mode == ProcessingInput) {
			data.resize(dim_info.size);
		}
		for(size_t i=0; i<dim_info.size; ++i) {
			(*this) << data[i];
		}
		(*this) << "info: " << info << Self::end;
	}

};





}


