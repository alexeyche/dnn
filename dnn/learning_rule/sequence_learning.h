#pragma once

#include "learning_rule.h"

#include <dnn/protos/sequence_learning.pb.h>
#include <dnn/protos/config.pb.h>
#include <ground/act_vector.h>
#include <dnn/neuron/spike_neuron_impl.h>
#include <dnn/neuron/spike_neuron_impl.h>
#include <dnn/neuron/defaults.h>
#include <dnn/reinforcement/sequence_likelihood.h>

namespace NDnn {

    struct TSequenceLearningConst: public IProtoSerial<NDnnProto::TSequenceLearningConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kSequenceLearningFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(TauFirstMoment);
            serial(TauSecondMoment);
        }

        double TauFirstMoment = 10.0;
        double TauSecondMoment = 100.0;
    };

    struct TSequenceLearningState: public IProtoSerial<NDnnProto::TSequenceLearningState>  {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kSequenceLearningStateFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(FirstMoment);
            serial(SecondMoment);
        }

        TActVector<double> FirstMoment;
        TActVector<double> SecondMoment;
    };

    template <template <typename> class TReinforcementType>
    struct TReinforcementAPI {

        template <typename TNeuron, typename TReinforcement>
        double GetTarget(const TTime& t, const TNeuron& neuron, const TReinforcement& /*reinforcement*/) {
            L_INFO << "Dealing with common get target";
            return neuron.Fired();
        }

        double GetModulation() const {
            return TGlobalCtx::Inst().GetRewardDelta();
        }
    };

    template <>
    struct TReinforcementAPI<TSequenceLikelihood> {
        template <typename TNeuron, typename TReinforcement>
        double GetTarget(const TTime& t, const TNeuron& /*neuron*/, const TReinforcement& reinforcement) {
            L_INFO << "Dealing with sequence likelihood";
            return reinforcement.GetTarget(t);
        }

        double GetModulation() const {
            return 1.0;
        }
    };


    template <typename TNeuron>
    class TSequenceLearning: public TLearningRule<TSequenceLearningConst, TSequenceLearningState, TNeuron> {
    public:
        using TPar = TLearningRule<TSequenceLearningConst, TSequenceLearningState, TNeuron>;

        void Reset() {
            TPar::s.FirstMoment.resize(TPar::GetSynapses().size(), 0.0);
            TPar::s.SecondMoment.resize(TPar::GetSynapses().size(), 0.0);
        }

        void PropagateSynapseSpike(const TSynSpike& sp) {
            TPar::s.FirstMoment.SetActive(sp.SynapseId);
        }

        void CalculateDynamics(const TTime& t) {
            auto& syns = TPar::GetMutSynapses();
            const auto& norm = TPar::Norm();
            const auto& neuron = TPar::Neuron();
            const auto& act = TPar::ActivationFunction();
            const double& p = neuron.SpikeProbability();
            const double p_deriv = act.SpikeProbabilityDerivative(neuron.Membrane());
            const double& M = neuron.ProbabilityModulation();

            const double fired = ReinforcementAPI.GetTarget(t, TPar::Neuron(), TPar::Reinforcement());

            double error = fired - p;
            const double modulation = ReinforcementAPI.GetModulation();

            auto synIdIt = TPar::s.FirstMoment.abegin();
            while (synIdIt != TPar::s.FirstMoment.aend()) {
                const ui32& synapseId = *synIdIt;

                auto& syn = syns.Get(synapseId);
                double& w = syn.MutWeight();
                double dw = (p_deriv/(p/M)) * syn.Potential() * error * modulation;

                // L_INFO << "(" << p_deriv << "/" << p/M << ") * " << syn.Potential() << " * (" << fired << " - " << p << ")" << " -> " << dw;

                // TPar::s.FirstMoment[synIdIt] += t.Dt * ( - TPar::s.FirstMoment[synIdIt] + dw)/TPar::c.TauFirstMoment;
                // TPar::s.SecondMoment.Get(synapseId) += t.Dt * ( - TPar::s.SecondMoment.Get(synapseId) + dw*dw)/TPar::c.TauSecondMoment;

                TPar::s.FirstMoment[synIdIt] = dw;

                // w += norm.Derivative(w,
                //     syn.LearningRate() * TPar::s.FirstMoment.Get(synapseId)/std::sqrt(TPar::s.SecondMoment.Get(synapseId) + 1e-08));
                w += norm.Derivative(w, syn.LearningRate() * TPar::s.FirstMoment[synIdIt]);

                // if ((std::fabs(TPar::s.FirstMoment[synIdIt]) < 1e-06)&&(std::fabs(TPar::s.SecondMoment.Get(synapseId)) < 1e-06)) {
                if (std::fabs(TPar::s.FirstMoment[synIdIt]) < 1e-06) {
                    TPar::s.FirstMoment.SetInactive(synIdIt);
                } else {
                    ++synIdIt;
                }
            }
        }

    private:
        TReinforcementAPI<TNeuron::TConfig::template TNeuronReinforcement> ReinforcementAPI;
    };

} // namespace NDnn
