#pragma once

#include <dnn/base/base.h>
#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/matrix.pb.h>

namespace NDnn {

    struct TDoubleMatrix : public IProtoSerial<NDnnProto::TDoubleMatrix> {
        TDoubleMatrix(ui32 _nrow, ui32 _ncol) {
            Allocate(_nrow, _ncol);
        }

        TDoubleMatrix() : NRow(0), NCol(0) {
        }
        
        TDoubleMatrix(const TVector<double> &v) {
        	Allocate(v.size(), 1);
        	for(ui32 i=0; i<v.size(); ++i) {
        		SetElement(i, 0, v[i]);
        	}
        }


        void SetLabel(ui32 i, string lab, TVector<ui32> &labels_ids) {
            ui32 lab_id;
            auto lab_ptr = std::find(UniqueLabels.begin(), UniqueLabels.end(), lab);
            if(lab_ptr == UniqueLabels.end()) {
                UniqueLabels.push_back(lab);
                lab_id = UniqueLabels.size()-1;
            } else {
                lab_id = lab_ptr - UniqueLabels.begin();
            }
            assert(lab_id < labels_ids.size());
            labels_ids[i] = lab_id;
        }


        void SetRowLabel(ui32 i, string lab) {
            SetLabel(i, lab, RowLabelsIds);
        }

        void SetColLabel(ui32 i, string lab) {
            SetLabel(i, lab, ColLabelsIds);
        }

        double& GetElement(ui32 i, ui32 j) {
            ENSURE((i<NRow) && (j<NCol),  "Out of bounds: " << i << "<" << NRow << " && " << j << "<" << NCol);
            
            return Values[j*NRow + i];
        }
        void SetElement(ui32 i, ui32 j, double val) {
            ENSURE((i<NRow) && (j<NCol),  "Out of bounds: " << i << "<" << NRow << " && " << j << "<" << NCol);
            
            Values[j*NRow + i] = val;
        }
        void Allocate(ui32 nr, ui32 nc) {
            if((NRow != 0)&&(NCol != 0)) {
                Values.clear();
            }
            NRow = nr;
            NCol = nc;
            Values.resize(NRow*NCol);
            RowLabelsIds.resize(NRow);
            ColLabelsIds.resize(NCol);
    	}
        void Fill(double val) {
    	    for(ui32 i=0; i<NRow; i++) {
        	    for(ui32 j=0; j<NCol; j++) {
            	    SetElement(i, j, val);
            	}
        	}
    	}
        
        void Norm() {
            for(ui32 i=0; i<RowSize(); ++i) {
                double acc = 0.0;
                for(ui32 j=0; j<ColSize(); ++j) {
                    acc += GetElement(i, j) * GetElement(i, j);
                }
                double n = sqrt(acc);
                for(ui32 j=0; j<ColSize(); ++j) {
                    SetElement(i, j, GetElement(i, j)/n);
                }
            }
        }
    	
        const ui32& ColSize() const { return NCol; }
    	const ui32& RowSize() const { return NRow; }

        const TVector<ui32>& GetColLabelsIds() const {
            return ColLabelsIds;
        }
        const TVector<ui32>& GetRowLabelsIds() const {
            return RowLabelsIds;
        }
        const TVector<string>& GetUniqueLabels() const {
            return UniqueLabels;
        }

        void TextRepr(std::ostream& o) {
            if(UniqueLabels.size()>0) {
                for(ui32 li=0; li<ColLabelsIds.size(); ++li) {
                    o << "\"" << "col." << li << ".lab." << UniqueLabels[ ColLabelsIds[li] ] << "\"";
                    if(li<ColLabelsIds.size()-1) o << " ";
                }
                o << "\n";
            }
            for(ui32 i=0; i<NRow; ++i) {
                if(UniqueLabels.size()>0) o << "\"" << "row." << i << ".lab." << UniqueLabels[ RowLabelsIds[i] ] << "\"" << " ";
                for(ui32 j=0; j<NCol; ++j) {
                    o << GetElement(i, j);
                    if(j<NCol-1) o << " ";
                }
                o << "\n";
            }
        }

        void SerialProcess(TProtoSerial& serial) override {
            serial(ColLabelsIds);
            serial(RowLabelsIds);
            serial(UniqueLabels);
            serial(NCol);
            serial(NRow);
            serial(Values);
        }

        double& operator () (ui32 i, ui32 j) {
            return GetElement(i, j);
        }

        double operator () (ui32 i, ui32 j) const {
            ENSURE((i<NRow) && (j<NCol),  "Out of bounds: " << i << "<" << NRow << " && " << j << "<" << NCol);
            
            return Values[j*NRow + i];
        }
        

    private:
        TVector<ui32> ColLabelsIds;
        TVector<ui32> RowLabelsIds;
        TVector<string> UniqueLabels;

    	ui32 NCol;
    	ui32 NRow;
    	TVector<double> Values;
    };



}
