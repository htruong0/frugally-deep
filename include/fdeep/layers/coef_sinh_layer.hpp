#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/recurrent_ops.hpp"
#include <fplus/fplus.hpp>

#include <limits>
#include <string>

namespace fdeep { namespace internal
{

class coef_sinh_layer : public activation_layer
{
public:
    explicit coef_sinh_layer(
        const std::string& name,
        float_type coef_scale)
        : activation_layer(name),
        coef_scale_(coef_scale)
    {
    }
protected:
    tensor transform_input(const tensor& in_vol) const override
    {
        return transform_tensor(
            fplus::multiply_with(coef_scale_),
            transform_tensor(sinh, in_vol)
        );
    }
    float_type coef_scale_;
};

} } // namespace fdeep, namespace internal
