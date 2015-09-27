#pragma once

#include <string>

using std::string;

#include <dnn/base/exceptions.h>
#include <dnn/contrib/spdlog/spdlog.h>


namespace dnn {


class Log {
public:
    Log()
    {
        log = spdlog::stdout_logger_mt("console");
        log->set_pattern("LOG [%H:%M:%S %z] [thread %t]: %v");
    }
    void setColors() {
        log->set_pattern("\x1b[32mLOG [%H:%M:%S %z] [thread %t]: %v\x1b[0m");
    }

    enum ELogLevel {INFO_LEVEL, DEBUG_LEVEL};

    void setLogLevel(ELogLevel lev) {
        switch(lev) {
            case Log::INFO_LEVEL:
                spdlog::set_level(spdlog::level::info);
                break;
            case Log::DEBUG_LEVEL:
                spdlog::set_level(spdlog::level::debug);
                break;
            default:
                throw dnnException() << "Invalig log level\n";
        }
    }
    ELogLevel getLogLevel() {
        if(log->level() == spdlog::level::info) {
            return INFO_LEVEL;
        }
        return DEBUG_LEVEL;
    }
    spdlog::details::line_logger info() {
        return log->info();
    }

    spdlog::details::line_logger debug() {
        return log->debug();
    }

    static Log& inst();
private:
    std::shared_ptr<spdlog::logger> log;
};


#define L_INFO \
    Log::inst().info()
#define L_DEBUG \
    Log::inst().debug()

}
