#pragma once

#include "time_series_info.h"

#include <dnn/util/pretty_print.h>
#include <dnn/util/ptr.h>
#include <dnn/util/maybe.h>
#include <dnn/base/base.h>

#include <numeric>

namespace NDnn {

    template <typename TDerived, typename TData>
    struct TTimeSeriesGeneric {
        using TSelf = TTimeSeriesGeneric<TDerived, TData>;

        TTimeSeriesGeneric() {}

        TTimeSeriesGeneric(const TVector<typename TData::TElement>& v) {
            Info.DimSize = 1;
            Data.resize(Info.DimSize);
            Data[0].Values = v;
        }

        void PadRightWithZeros(ui32 padSize) {
            for(auto &v: Data) {
                for(ui32 zi=0; zi<padSize; ++zi) {
                    v.Values.push_back(typename TData::TElement(0.0));
                }
            }
        }

        void CutFromRight(ui32 cutSize) {
            for(auto &v: Data) {
                for(ui32 zi=0; zi<cutSize; ++zi) {
                    v.Values.pop_back();
                }
            }
        }

        void AssertAnotherTs(const TDerived& anotherTs) {
            ENSURE((Dim() == anotherTs.Dim()) && Length() == anotherTs.Length(), "Can't work with time series with different dimenstions");
        }

        void Normalize() {
            for(ui32 di=0; di < Data.size(); ++di) {
                double acc(0.0);
                for(const auto &v: Data[di].Values) {
                    acc += v*v;
                }
                acc = sqrt(acc);
                for(auto &v: Data[di].Values) {
                    v = v / acc;
                }
            }
        }

        double InnerProductMul(const TDerived& anotherTs) {
            AssertAnotherTs(anotherTs);
            double acc = 1.0;
            for(ui32 di=0; di < Data.size(); ++di) {
                acc *= std::inner_product(Data[di].Values.begin(), Data[di].Values.end(), anotherTs.Data[di].Values.begin(), 0.0);
            }
            return acc;
        }

        double InnerProductAcc(const TDerived& anotherTs) {
            AssertAnotherTs(anotherTs);
            double acc = 0.0;
            for(ui32 di=0; di < Data.size(); ++di) {
                acc += std::inner_product(Data[di].Values.begin(), Data[di].Values.end(), anotherTs.Data[di].Values.begin(), 0.0)/Data[di].Values.size();
            }
            return acc;
        }

        void InnerProduct(const TDerived& anotherTs) {
            ENSURE( (Dim() == anotherTs.Dim()) && (anotherTs.Dim() != 1) && (Length() == anotherTs.Length()), "Can't multiply time series with different dimensions or length");
            for(ui32 di=0; di < Data.size(); ++di) {
                for(ui32 vi=0; vi < Data[di].Values.size(); ++vi) {
                    ui32 another_ts_di = di;
                    if(anotherTs.Dim() == 1) {
                        another_ts_di = 0;
                    }
                    Data[di].Values[vi] *= anotherTs.Data[another_ts_di].Values[vi];
                }
            }
        }

        TVector<typename TData::TElement>& GetMutVector(ui32 ndim) {
            while(ndim >= Info.DimSize) {
                Info.DimSize = ndim+1;
                Data.push_back(TData());
            }
            assert(Info.DimSize == Data.size());
            return Data[ndim].Values;
        }

        const TVector<typename TData::TElement>& GetVector(ui32 ndim) const {
            return const_cast<TSelf*>(this)->GetMutVector(ndim);
        }

        TVector<typename TData::TElement> GetColumnVector(ui32 xi) {
            TVector<typename TData::TElement> col(Dim());
            for(ui32 di=0; di < Dim(); ++di) {
                assert(xi<Data[di].Values.size());
                col[di] = Data[di].Values[xi];
            }
            return col;
        }

        void AddValue(ui32 dim_index, typename TData::TElement val) {
            if(dim_index == Info.DimSize) {
                Info.DimSize = dim_index+1;
                Data.push_back(TData());
                assert(Info.DimSize == Data.size());
            }
            ENSURE(dim_index < Data.size(), "Trying to make write outside of the memory");

            Data[dim_index].Values.push_back(val);
        }
        ui32 Length() const {
            if(Data.size() == 0) return 0;
            return Data[0].Values.size();
        }

        ui32 Dim() const {
            return Data.size();
        }

        const typename TData::TElement& GetValueAt(const ui32 &index) const {
            return Data[0].Values[index];
        }
        const typename TData::TElement& GetValueAtDim(const ui32 &index, const ui32 &dim) const {
            return Data[dim].Values[index];
        }
        const typename TData::TElement& operator () (const ui32 &dim, const ui32 &index) const {
            return Data[dim].Values[index];
        }

        void AssertOneLabel() {
            ENSURE(Info.UniqueLabels.size() != 0, "Trying to get one label from nonlabeled time series");
            ENSURE(Info.UniqueLabels.size() == 1, "Trying to get one label from multilabeled time series");
        }

        const string& GetLabel() {
            AssertOneLabel();
            return Info.UniqueLabels[0].Name;
        }

        const ui32& GetLabelId() {
            AssertOneLabel();
            return Info.LabelsStart[0].LabelId;
        }

        void SetDimSize(const ui32 &size) {
            while(Info.DimSize < size) {
                Data.push_back(TData());
                Info.DimSize = Data.size();
            }
        }
        const double& GetTimeDelta() const {
            return Info.Dt;
        }

        void ChangeTimeDelta(const double &dt) {
            Info.ChangeTimeDelta(dt);
        }

        TVector<TDerived> Chop()  {
            ui32 elem_id = 0;
            TVector<TDerived> ts_chopped;

            for(ui32 li=0; li<Info.LabelsStart.size(); ++li) {
                const ui32 &start_of_label = Info.LabelsStart[li].Start;
                const ui32 &label_id = Info.LabelsStart[li].LabelId;

                const ui32 &end_of_label = start_of_label + Info.UniqueLabels[label_id].Duration;
                const string &label = Info.UniqueLabels[label_id].Name;

                TDerived labeled_ts;
                for(; elem_id < end_of_label; ++elem_id) {
                    for(ui32 di=0; di<Data.size(); ++di) {
                        labeled_ts.AddValue(di, Data[di].Values[elem_id]);
                    }
                }
                labeled_ts.Info.AddLabelAtPos(label, 0, labeled_ts.Length());
                ts_chopped.push_back(labeled_ts);
            }
            L_DEBUG << "TimeSeries, Successfully chopped time series in " << ts_chopped.size() << " chunks";
            return ts_chopped;
        }

        void MultiplyOnDimensions(ui32 dims) {
            ENSURE(Dim() == 1, "Can't multiply non one dimensional time series");
            Data.resize(dims, Data[0]);
            Info.DimSize = Data.size();
        }

        void SerialProcess(TProtoSerial& serial) {
            serial(Info);
            serial(Data);
        }



        TTimeSeriesInfo Info;
        TVector<TData> Data;
    };

}

