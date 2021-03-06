#pragma once

#include <condition_variable>

#include <ground/base/base.h>
#include <ground/server/server.h>
#include <ground/ts/time_series.h>


namespace NDnn {
	using namespace NGround;
	
	class TDispatcher {
	public:
		TDispatcher();
		
		TDispatcher(const TDispatcher& other);

		TDispatcher& operator =(const TDispatcher& other);

		void SetPort(ui32 port);
		
		const ui32& GetPort() const;

		double GetNeuronInput(ui32 layerId, ui32 neuronId);

		void MainLoop();

		void SetInputTimeSeries(const TTimeSeries&& ts);

		void ShutDown();

	private:
		std::condition_variable InputDataIsReady;
		bool InputDataIsReadyVar;
		std::mutex	InputDataMutex;

		TTimeSeries InputData;

		TServer Server;
		TVector<ui32> InputDataIdx;
	};


} // namspace NDnn