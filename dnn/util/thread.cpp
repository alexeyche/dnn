#include "thread.h"

#include <dnn/util/log/log.h>

#include <cmath>

namespace NDnn {

    TVector<TIndexSlice> DispatchOnThreads(ui32 elements_size, ui32 jobs) {
        TVector<TIndexSlice> out;
        double el_per_thread = static_cast<double>(elements_size) / jobs;

        for(ui32 ji=0; ji < jobs; ji++) {
            ui32 first = std::min( static_cast<ui32>(std::floor(ji * el_per_thread)), elements_size );
            ui32 last  = std::min( static_cast<ui32>(std::floor((ji+1) * el_per_thread)), elements_size );
            
            out.push_back( TIndexSlice(first, last) );
        }
        return out;
    }

} // namespace NDnn