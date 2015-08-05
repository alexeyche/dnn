#include "log.h"

namespace dnn {


Log& Log::inst() {
    static Log _inst;
    return _inst;
}


}
