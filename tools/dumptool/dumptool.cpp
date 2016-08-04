
#include <iostream>
#include <fstream>

#include <dnn/protos/options.pb.h>
#include <dnn/protos/config.pb.h>
#include <ground/proto_options.h>
#include <ground/ts/time_series.h>
#include <ground/ts/spikes_list.h>
#include <ground/serial/bin_serial.h>
#include <ground/log/log.h>
#include <ground/stat_gatherer.h>

using namespace NGround;

#include "dumptool.h"


int main(int argc, const char** argv) {
	TProtoOptions<NDnnProto::TDumptoolOptions> clOptions(argc, argv, "Dynamic neural network dumptool");

    NDnnProto::TDumptoolOptions options;
    if (!clOptions.Parse(options)) {
        return 0;
    }
    
    TLog::Instance().SetLogLevel(TLog::DEBUG_LEVEL);

    std::ifstream input(options.input(), std::ios::binary);
    TBinSerial serial(input);
    switch (serial.ReadProtobufType()) {
        case EProto::TIME_SERIES:
            DumpEntity<TTimeSeries>(serial);
            break;
        case EProto::SPIKES_LIST:
            DumpEntity<TSpikesList>(serial);
            break;
        case EProto::STATISTICS:
            DumpEntities<TStatistics>(serial);
            break;
        case EProto::CONFIG:
            {
                NDnnProto::TConfig config;
                serial.ReadProtobufMessage(config);
                std::cout << config.DebugString();
            }
            break;
        default:
            throw TErrException() << "Failed to recognize protobuf type";
    }
    return 0;
}
