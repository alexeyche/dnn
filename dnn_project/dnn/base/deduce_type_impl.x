#define REG_TYPE(name) \
    if (info == typeid(name)) { \
        return #name; \
    } \

#define REG_TYPE_WITH_CONST(name) \
    REG_TYPE(name) \
    if (info == typeid(name##C)) { \
        return string(#name) + string("C"); \
    } \

#define REG_TYPE_WITH_CONST_AND_STATE(name) \
    REG_TYPE(name)\
    REG_TYPE_WITH_CONST(name)\
    if (info == typeid(name##State)) { \
        return string(#name) + string("State"); \
    } \

#include REG_FILE

#undef REG_TYPE
#undef REG_TYPE_WITH_CONST
#undef REG_TYPE_WITH_CONST_AND_STATE

//L_DEBUG << "Failed to deduce type for: " << info.name();
return string("");