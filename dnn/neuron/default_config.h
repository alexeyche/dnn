#pragma once

#include "config.h"

#include <dnn/synapse/basic_synapse.h>
#include <dnn/activation/determ.h>
#include <dnn/receptive_field/ident.h>



namespace NDnn {

	using TDefaultConfig = TNeuronConfig<TBasicSynapse, TDeterm, TNoInput>;

} // namespace NDnn
