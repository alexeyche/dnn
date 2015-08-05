#pragma once

#include <vector>

using std::vector;

#include <dnn/io/serialize.h>
#include <dnn/protos/matrix.pb.h>

namespace dnn {




/*@GENERATE_PROTO@*/
class DoubleMatrix : public Serializable<Protos::DoubleMatrix> {
    void setLabel(size_t i, string lab, vector<size_t> &labels_ids) {
        size_t lab_id;
        auto lab_ptr = std::find(unique_labels.begin(), unique_labels.end(), lab);
        if(lab_ptr == unique_labels.end()) {
            unique_labels.push_back(lab);
            lab_id = unique_labels.size()-1;
        } else {
            lab_id = lab_ptr - unique_labels.begin();
        }
        assert(lab_id < labels_ids.size());
        labels_ids[i] = lab_id;
    }
public:
    void serial_process() {
    	begin() << "ncol_v: " << ncol_v << ", "
    			<< "nrow_v: " << nrow_v << ", "
    			<< "values: " << values << ", ";

        (*this) << "unique_labels: " << unique_labels << ", ";
        if (unique_labels.size()>0) {
            (*this) << "row_labels_ids: " << row_labels_ids << ", ";
            (*this) << "col_labels_ids: " << col_labels_ids << ", ";
        }
        (*this) << Self::end;
    }
    double& operator () (size_t i, size_t j) {
        assert( (nrow_v != 0) && (ncol_v != 0) );
        if( ! ( (i<nrow_v) && (j<ncol_v) ) ) {
            throw dnnException() << "assert: " << i << "<" << nrow_v << " && " << j << "<" << ncol_v << "\n";
        }
        return values[j*nrow_v + i];
    }

    double operator () (size_t i, size_t j) const {
        assert( (nrow_v != 0) && (ncol_v != 0) );
        if( ! ( (i<nrow_v) && (j<ncol_v) ) ) {
            throw dnnException() << "assert: " << i << "<" << nrow_v << " && " << j << "<" << ncol_v << "\n";
        }
        return values[j*nrow_v + i];
    }
    DoubleMatrix(size_t _nrow, size_t _ncol) {
        allocate(_nrow, _ncol);
    }

    DoubleMatrix() : nrow_v(0), ncol_v(0) {
    }
    DoubleMatrix(const vector<double> &v) {
    	allocate(v.size(), 1);
    	for(size_t i=0; i<v.size(); ++i) {
    		setElement(i, 0, v[i]);
    	}
    }


    void setRowLabel(size_t i, string lab) {
        setLabel(i, lab, row_labels_ids);
    }

    void setColLabel(size_t i, string lab) {
        setLabel(i, lab, col_labels_ids);
    }
    double getElement(size_t i, size_t j) const {
        assert( (nrow_v != 0) && (ncol_v != 0) );
        if( ! ( (i<nrow_v) && (j<ncol_v) ) ) {
            throw dnnException() << "assert: " << i << "<" << nrow_v << " && " << j << "<" << ncol_v << "\n";
        }
        return values[j*nrow_v + i];
    }
    void setElement(size_t i, size_t j, double val) {
        assert( (nrow_v != 0) && (ncol_v != 0) );
        if( ! ( (i<nrow_v) && (j<ncol_v) ) ) {
            throw dnnException() << "assert: " << i << "<" << nrow_v << " && " << j << "<" << ncol_v << "\n";
        }
        values[j*nrow_v + i] = val;
    }
    void allocate(size_t nr, size_t nc) {
        if((nrow_v != 0)&&(ncol_v != 0)) values.clear();
        nrow_v = nr;
        ncol_v = nc;
        values.resize(nrow_v*ncol_v);
        row_labels_ids.resize(nrow_v);
        col_labels_ids.resize(ncol_v);
	}
    void fill(double val) {
	    for(size_t i=0; i<nrow_v; i++) {
    	    for(size_t j=0; j<ncol_v; j++) {
        	    setElement(i, j, val);
        	}
    	}
	}
    void norm() {
        for(size_t i=0; i<nrow(); ++i) {
            double acc = 0.0;
            for(size_t j=0; j<ncol(); ++j) {
                acc += getElement(i, j) * getElement(i, j);
            }
            double n = sqrt(acc);
            for(size_t j=0; j<ncol(); ++j) {
                setElement(i, j, getElement(i, j)/n);
            }
        }
    }
	inline const size_t& ncol() const { return ncol_v; }
	inline const size_t& nrow() const { return nrow_v; }

    const vector<size_t>& colLabelsIds() {
        return col_labels_ids;
    }
    const vector<size_t>& rowLabelsIds() {
        return row_labels_ids;
    }
    const vector<string>& uniqueLabels() {
        return unique_labels;
    }

    void textRepr(ostream &o) {
        if(unique_labels.size()>0) {
            for(size_t li=0; li<col_labels_ids.size(); ++li) {
                o << "\"" << "col." << li << ".lab." << unique_labels[ col_labels_ids[li] ] << "\"";
                if(li<col_labels_ids.size()-1) o << " ";
            }
            o << "\n";
        }
        for(size_t i=0; i<nrow_v; ++i) {
            if(unique_labels.size()>0) o << "\"" << "row." << i << ".lab." << unique_labels[ row_labels_ids[i] ] << "\"" << " ";
            for(size_t j=0; j<ncol_v; ++j) {
                o << getElement(i, j);
                if(j<ncol_v-1) o << " ";
            }
            o << "\n";
        }
    }

private:
    vector<size_t> col_labels_ids;
    vector<size_t> row_labels_ids;
    vector<string> unique_labels;

	size_t ncol_v;
	size_t nrow_v;
	vector<double> values;
};



}
