#define REG_TYPE(name) \
    Factory::registerType<name>(#name);\

#define REG_TYPE_WITH_CONST(name) \
    REG_TYPE(name) \
    Factory::registerType<name##C>(string(#name) + string("C"));\

#define REG_TYPE_WITH_CONST_AND_STATE(name) \
    REG_TYPE(name)\
    REG_TYPE_WITH_CONST(name)\
    Factory::registerType<name##State>(string(#name) + string("State"));\

#include REG_FILE

#undef REG_TYPE
#undef REG_TYPE_WITH_CONST
#undef REG_TYPE_WITH_CONST_AND_STATE