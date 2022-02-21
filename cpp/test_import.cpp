#include <fdeep/fdeep.hpp>
int main()
{
    // load model from Keras
    const auto model = fdeep::load_model("test_export.json");
    // test phase-space point
    fdeep::tensor ps(fdeep::tensor_shape(5, 4), 0.0);
    ps.set(fdeep::tensor_pos(0, 0), 106.49025546);
    ps.set(fdeep::tensor_pos(0, 1), -96.79583241);
    ps.set(fdeep::tensor_pos(0, 2), 40.39606627);
    ps.set(fdeep::tensor_pos(0, 3), 18.40921418);
    ps.set(fdeep::tensor_pos(1, 0), 369.24318288);
    ps.set(fdeep::tensor_pos(1, 1), -242.84506655);
    ps.set(fdeep::tensor_pos(1, 2), -33.88003621);
    ps.set(fdeep::tensor_pos(1, 3), 276.07778777);
    ps.set(fdeep::tensor_pos(2, 0), 242.36511604);
    ps.set(fdeep::tensor_pos(2, 1), 181.98909365);
    ps.set(fdeep::tensor_pos(2, 2), 64.41873357);
    ps.set(fdeep::tensor_pos(2, 3), -146.53001751);
    ps.set(fdeep::tensor_pos(3, 0), 71.92597541);
    ps.set(fdeep::tensor_pos(3, 1), -27.74400632);
    ps.set(fdeep::tensor_pos(3, 2), -35.89323424);
    ps.set(fdeep::tensor_pos(3, 3), -55.814799);
    ps.set(fdeep::tensor_pos(4, 0), 209.97547021);
    ps.set(fdeep::tensor_pos(4, 1), 185.39581163);
    ps.set(fdeep::tensor_pos(4, 2), -35.04152939);
    ps.set(fdeep::tensor_pos(4, 3), -92.14218543);
    // corresponding recoil factors
    std::vector<float> y = {0.18763366, 0.64391875, 0.46874577, 0.13910656, 0.11024843,
        0.11514172, 0.21459483, 0.46763787, 0.6709091 , 0.7601958 ,
        0.77691776, 0.5358791 , 0.634991  , 0.14987242, 0.17472999,
        0.72559685, 0.4461702 , 0.7541445 , 0.24583283, 0.07320979,
        0.4568705 , 0.06361642, 0.01795081, 0.16241129, 0.21394919,
        0.07112551, 0.3807182};
    const fdeep::tensor ys = fdeep::tensor(fdeep::tensor_shape(27), y);
    std::cout << fdeep::show_tensor(ps) << std::endl;
    std::cout << fdeep::show_tensor(ys) << std::endl;
    std::cout << fdeep::show_tensor_shape(ps.shape()) << std::endl;
    std::cout << fdeep::show_tensor_shape(ys.shape()) << std::endl;
    // predict C_ijk
    const fdeep::tensors result = model.predict({ps, ys});

    // check the same as Python output:
    // [ 3.86544478e-18,  2.07138251e-17,  1.56155636e-17,  4.03263535e-18,
    //     5.07069243e-18,  9.85903834e-18,  1.72153210e-17,  3.39332681e-17,
    //     3.84675381e-18,  4.26129330e-18,  1.78582405e-17,  1.47812378e-17,
    //     8.86741649e-19,  1.34045991e-17,  9.52118634e-18,  1.73699061e-17,
    //     2.16201239e-17, -1.18532720e-19,  3.49030927e-18,  1.07555044e-17,
    //     5.88288370e-18,  9.23143903e-18,  1.26430578e-17,  8.03901930e-18,
    //     4.92265977e-18,  1.34006948e-17,  5.48384929e-18,  1.02483811e-18,
    //     6.30411343e-18,  7.06491652e-19, -1.57254414e-19, -1.90148518e-19,
    //    -9.68044172e-20]

    const std::vector<float> res = single_tensor_from_tensors(result).to_vector();
    for (float i: res)
        std::cout << i << ',';

    // need to multiply with D_ij,k to reproduce matrix element
}
