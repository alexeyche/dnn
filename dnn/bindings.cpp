#include "bindings.h"

#include <ground/log/log.h>
#include <ground/string.h>
#include <ground/ts/time_series.h>
#include <ground/protos/time_series.pb.h>
#include <ground/serial/bin_serial.h>

#include <iostream>
#include <fstream>

using namespace NGround;
namespace NGrPb = NGroundProto;

void write_time_series(const double* data, int nrows, int ncols, const char* label, const char* dst_file) {
	L_INFO << "Writing TimeSeries of size " << nrows << "x" << ncols << " into " << dst_file;
	NGrPb::TTimeSeries ts;
	
	auto* info = ts.mutable_info();
	info->set_dimsize(nrows);
	info->set_dt(1.0);	
	auto* lab = info->add_uniquelabels();
	lab->set_name(label);
	lab->set_duration(ncols);
	auto* lab_start = info->add_labelsstart();
	lab_start->set_start(0);
	lab_start->set_labelid(0);

	for (ui32 rid = 0; rid < nrows; ++rid) {
		NGrPb::TTimeSeriesData* ts_data = ts.add_data();
		for (ui32 cid = 0; cid < ncols; ++cid) {
			ts_data->add_values(data[rid * ncols + cid]);
		}
	}
	std::ofstream output(dst_file, std::ios::binary);
    TBinSerial(output).WriteProtobuf<TTimeSeries>(ts);
}