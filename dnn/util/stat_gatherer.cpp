#include "stat_gatherer.h"

#include <dnn/util/serial/bin_serial.h>

#include <fstream>

namespace NDnn {
	
	TStatGatherer::TStatGatherer() {
	}

	TStatGatherer::TStatGatherer(const TStatGatherer& other) {
		(*this) = other;	
	}

	TStatGatherer& TStatGatherer::operator=(const TStatGatherer& other) {
		if (this != &other) {
			Stats = other.Stats;
			ActiveStats.clear();
			for (auto& stat: Stats) {
				ActiveStats.push_back(&stat);
			}
		}
		return *this;
	}

	void TStatGatherer::Init() {
		ActiveStats.clear();
		for (auto& stat: Stats) {
			ActiveStats.push_back(&stat);
		}
		std::sort(ActiveStats.begin(), ActiveStats.end(), 
			[](const TPtr<TStatCollector>& left, const TPtr<TStatCollector>& right) -> bool {
				return left->GetFrom() > right->GetFrom();
			}
		);
	}

	void TStatGatherer::ListenStat(const TString& name, std::function<double()> cb, ui32 from, ui32 to) {
		L_DEBUG << "Listening statistics " << name << " from " << from  << " to " << to; 
		Stats.emplace_back(name, cb, from, to);
	}

	void TStatGatherer::Collect(const TTime& t) {
		auto statIt = ActiveStats.begin();
		while (statIt != ActiveStats.end()) {
			if (t.T > statIt->Get()->GetTo()) {
				statIt = ActiveStats.erase(statIt);
			} else {
				if (t.T >= statIt->Get()->GetFrom()) {
					statIt->Get()->Collect();
				}
				++statIt;	
			}
		}
	}

	void TStatGatherer::SaveStat(const TString& fname) {
		std::ofstream output(fname, std::ios::binary);
	    TBinSerial serial(output);
	    for (auto& stat: Stats) {
	    	serial.WriteObject<TStatistics>(stat.GetMutStatistics());
	    }
	}

	ui32 TStatGatherer::Size() const {
		return Stats.size();
	}

} // namespace NDnn