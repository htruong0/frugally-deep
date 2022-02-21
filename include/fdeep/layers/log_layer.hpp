#pragma once

#include "fdeep/layers/activation_layer.hpp"
#include "fdeep/recurrent_ops.hpp"

#include <limits>
#include <string>

namespace fdeep { namespace internal
{

class log_layer : public activation_layer
{
public:
    explicit log_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    tensor transform_input(const tensor& in_vol) const override
    {
        return transform_tensor(logarithm, in_vol);
    }
};

} } // namespace fdeep, namespace internal
