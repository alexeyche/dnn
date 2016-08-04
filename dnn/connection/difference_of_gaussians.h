#pragma once

#include <dnn/connection/connection.h>
#include <dnn/protos/difference_of_gaussians.pb.h>

#include <dnn/neuron/spike_neuron_impl.h>

namespace NDnn {

    struct TDifferenceOfGaussiansConst : public IProtoSerial<NDnnProto::TDifferenceOfGaussiansConst> {
        static const auto ProtoFieldNumber = NDnnProto::TConnection::kDifferenceOfGaussiansFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Dimension);
            serial(SigmaPos);
            serial(SigmaNeg);
            serial(AmpNeg);
            serial(AmpPos);
            serial(ApplyAmplitude);
        }

        int Dimension = 1;
        double SigmaPos = 3.0;
        double SigmaNeg = 15.0;
        double AmpNeg = 3.0;
        double AmpPos = 1.0;
        bool ApplyAmplitude = true;
    };


    class TDifferenceOfGaussians : public TConnection<TDifferenceOfGaussiansConst> {
    public:
        static double GaussFunction2d(double x, double y, double xc, double yc, double sigma) {
            return (1/(sqrt(2*PI)*sigma)) * exp( - ( (x-xc)*(x-xc) + (y-yc)*(y-yc) )/(2*sigma*sigma) ) ;
        }
        static double GaussFunction(double x, double xc, double sigma) {
            return (1/(sqrt(2*PI)*sigma)) * exp( - ( (x-xc)*(x-xc) )/(2*sigma*sigma) ) ;
        }

        TConnectionRecipe GetConnectionRecipe(const TNeuronSpaceInfo& left, const TNeuronSpaceInfo& right) override final {
            TConnectionRecipe recipe;

            double v = 0.0;
            if(c.Dimension == 1) {
                int right_circled = right.LocalId-right.LayerSize;
                int left_circled = left.LocalId-left.LayerSize;
                v += c.AmpPos * GaussFunction(left.LocalId, right.LocalId, c.SigmaPos) -
                          c.AmpNeg * GaussFunction(left.LocalId, right.LocalId, c.SigmaNeg);
                v += c.AmpPos * GaussFunction(left.LocalId, right_circled, c.SigmaPos) -
                          c.AmpNeg * GaussFunction(left.LocalId, right_circled, c.SigmaNeg);
                v += c.AmpPos * GaussFunction(left_circled, right.LocalId, c.SigmaPos) -
                          c.AmpNeg * GaussFunction(left_circled, right.LocalId, c.SigmaNeg);

            } else
            if(c.Dimension == 2) {
                int right_xi_circled = right.RowId-right.ColumnSize;
                int left_xi_circled = left.RowId-left.ColumnSize;
                int right_yi_circled = right.ColId-right.ColumnSize;
                int left_yi_circled = left.ColId-left.ColumnSize;

                v += c.AmpPos * GaussFunction2d(right.RowId, right.ColId, left.RowId, left.ColId, c.SigmaPos) -
                          c.AmpNeg * GaussFunction2d(right.RowId, right.ColId, left.RowId, left.ColId, c.SigmaNeg);
                v += c.AmpPos * GaussFunction2d(right.RowId, right.ColId, left_xi_circled, left_yi_circled, c.SigmaPos) -
                          c.AmpNeg * GaussFunction2d(right.RowId, right.ColId, left_xi_circled, left_yi_circled, c.SigmaNeg);
                v += c.AmpPos * GaussFunction2d(right_xi_circled, right_yi_circled, left.RowId, left.ColId, c.SigmaPos) -
                          c.AmpNeg * GaussFunction2d(right_xi_circled, right_yi_circled, left.RowId, left.ColId, c.SigmaNeg);
            } else {
                throw TErrException() << "Can't build DifferenceOfGaussians with dimension like this: " << c.Dimension << "\n"
                                      << "Only 1 or 2 dimension supported\n";
            }
            if (fabs(v) > 1e-05) {
                recipe.Exists = true;
                recipe.Amplitude = v;
            }
            return recipe;
        }
    };





}; // namespace NDnn