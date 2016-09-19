// #pragma once

// #include "cost_function.h"

// #include <dnn/protos/regression.pb.h>
// #include <dnn/protos/config.pb.h>
// #include <dnn/sim/global_ctx.h>

// namespace NDnn {

//     struct TRegressionCostConst: public IProtoSerial<NDnnProto::TRegressionCostConst> {
//         static const auto ProtoFieldNumber = NDnnProto::TLayer::kRegressionCostFieldNumber;

//         void SerialProcess(TProtoSerial& serial) {
//             serial(LearningRate);
//         }


//         double LearningRate = 1e-03;
//     };

//     template <typename TNeuron>
//     class TRegressionCost: public TCostFunction<TRegressionCostConst, TNeuron> {
//     public:
//         using TPar = TCostFunction<TRegressionCostConst, TNeuron>;

//         void CalculateError() {
//             const auto& neuron = TPar::Neuron();
//             const auto& act = TPar::ActivationFunction();

//             const double& p = neuron.SpikeProbability();
//             const double& M = neuron.ProbabilityModulation();
//             const double p_deriv = act.SpikeProbabilityDerivative(neuron.Membrane());
//             const double fired = static_cast<double>(neuron.Fired());

//             if (Index >= Seq.size()) {
//                 Error = p; //TPar::Neuron().Membrane(); //TPar::Neuron().Membrane();
//             } else {
//                 Error = p - Seq[++Index];
//             }

            

//             for (auto& syn: TPar::MutSynapses()) {
//                 double dw = - (p_deriv/(p/M)) * syn.Potential() * Error;

//                 // double dy = (p_deriv/p) * syn.Potential() * (Error -  p);
//                 syn.MutWeight() += TPar::c.LearningRate * dw; 
//             }

//             // TPar::Neuron().MutMembrane() = std::fabs(Error);

//             TGlobalCtx::Inst().SetError(TPar::SpaceInfo().GlobalId, Error*Error);
//         }

//         void SetTarget(const TVector<double>& seq) {
//             Seq = seq;
//         }

//         const double& GetError() const {
//             return Error;
//         }

//     private:
//         double Error = 0.0;
//         TVector<double> Seq;
//         ui32 Index = 0;

//         TActVector<double> FirstMoment;
//         TActVector<double> SecondMoment;
//     };

// } // namespace NDnn
