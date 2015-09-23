#define REG_TYPE(name) \
    class name; \

#define REG_TYPE_WITH_CONST(name) \
    REG_TYPE(name) \
    class name##C; \

#define REG_TYPE_WITH_CONST_AND_STATE(name) \
    REG_TYPE(name) \
    REG_TYPE_WITH_CONST(name) \
    class name##State; \

#include REG_FILE

#undef REG_TYPE
#undef REG_TYPE_WITH_CONST
#undef REG_TYPE_WITH_CONST_AND_STATE