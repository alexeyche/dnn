#pragma once

#include <atomic>

#include <dnn/protos/reward_control.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/util/statistics.h>

namespace dnn {



/*@GENERATE_PROTO@*/
struct RewardControlState : public Serializable<Protos::RewardControlState> {
    RewardControlState()
    : R(0.0)
    , meanR(0.0) {}

    void serial_process() {
        begin() << "R: "     << R << ", "
                << "meanR: " << meanR << Self::end;
    }

    double R;
    double meanR;
};

/*@GENERATE_PROTO@*/
struct RewardControlC : public Serializable<Protos::RewardControlC> {
    RewardControlC()
    : tau_trace(100.0)
    , tau_mean_trace(10000.0) {}

    void serial_process() {
        begin() << "tau_trace: "      << tau_trace << ", "
                << "tau_mean_trace: " << tau_mean_trace << Self::end;
    }


    double tau_trace;
    double tau_mean_trace;
};


class RewardControl : public SerializableBase {
public:
    const string name() const { return "RewardControl"; }

    RewardControl() : gathered_reward(0.0) {}

    void calculateDynamics(const Time& t) {
        s.meanR += t.dt * ( - (s.meanR - s.R)/c.tau_mean_trace );
        s.R += t.dt * ( - (s.R - gathered_reward.load())/c.tau_trace );

        gathered_reward.store(0.0);

        stat.add("meanR", s.meanR);
        stat.add("R", s.R);
    }

    void serial_process() {
        begin() << "Constants: " << c;

        if (messages->size() == 0) {
            (*this) << Self::end;
            return;
        }

        (*this) << "State: " << s << Self::end;
    }

    void gatherReward(const double &R) {
        atomicDoubleAdd(gathered_reward, R);
    }

    Statistics& getStat() {
        return stat;
    }

    RewardControl& operator = (const RewardControl &other) {
        gathered_reward.store(other.gathered_reward.load());
        s = other.s;
        c = other.c;
        stat = other.stat;
        return *this;
    }
    RewardControl(const RewardControl &other) {
        gathered_reward.store(other.gathered_reward.load());
        s = other.s;
        c = other.c;
        stat = other.stat;
    }

private:
    std::atomic<double> gathered_reward;

    RewardControlState s;
    RewardControlC c;

    Statistics stat;
};




}