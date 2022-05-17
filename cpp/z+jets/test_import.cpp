// need to use doubles instead of floats
#define FDEEP_FLOAT_TYPE double 
#include <fdeep/fdeep.hpp>
int main()
{
    // load model from Keras
    const auto model = fdeep::load_model("test_zjets_model.json");
    // test phase-space point
    fdeep::tensor ps(fdeep::tensor_shape(8, 4), 0.0);
    ps.set(fdeep::tensor_pos(0, 0),  359.2437206134463);
    ps.set(fdeep::tensor_pos(0, 1),  0.0);
    ps.set(fdeep::tensor_pos(0, 2),  0.0);
    ps.set(fdeep::tensor_pos(0, 3),  359.24372061344627);
    ps.set(fdeep::tensor_pos(1, 0),  40.09826410192703);
    ps.set(fdeep::tensor_pos(1, 1),  0.0);
    ps.set(fdeep::tensor_pos(1, 2),  0.0);
    ps.set(fdeep::tensor_pos(1, 3),  -40.09826410192697);
    ps.set(fdeep::tensor_pos(2, 0),  193.28144526790766);
    ps.set(fdeep::tensor_pos(2, 1),  -20.089738939578467);
    ps.set(fdeep::tensor_pos(2, 2),  30.322400322027168);
    ps.set(fdeep::tensor_pos(2, 3),  189.82800508065583);
    ps.set(fdeep::tensor_pos(3, 0),  17.42990734238425);
    ps.set(fdeep::tensor_pos(3, 1),  -0.4276531958676415);
    ps.set(fdeep::tensor_pos(3, 2),  15.97282721908709);
    ps.set(fdeep::tensor_pos(3, 3),  -6.963301899053043);
    ps.set(fdeep::tensor_pos(4, 0),  28.006033586507346);
    ps.set(fdeep::tensor_pos(4, 1),  0.1808385647842115);
    ps.set(fdeep::tensor_pos(4, 2),  -20.15530230326735);
    ps.set(fdeep::tensor_pos(4, 3),  19.443996598589763);
    ps.set(fdeep::tensor_pos(5, 0),  39.303333683100966);
    ps.set(fdeep::tensor_pos(5, 1),  -4.795291163542301);
    ps.set(fdeep::tensor_pos(5, 2),  31.606603308033982);
    ps.set(fdeep::tensor_pos(5, 3),  22.86437947092829);
    ps.set(fdeep::tensor_pos(6, 0),  74.1937758382832);
    ps.set(fdeep::tensor_pos(6, 1),  36.12381460962831);
    ps.set(fdeep::tensor_pos(6, 2),  -37.47340228685164);
    ps.set(fdeep::tensor_pos(6, 3),  52.87277666472987);
    ps.set(fdeep::tensor_pos(7, 0),  47.1274889971899);
    ps.set(fdeep::tensor_pos(7, 1),  -10.991969875424113);
    ps.set(fdeep::tensor_pos(7, 2),  -20.273126259029254);
    ps.set(fdeep::tensor_pos(7, 3),  41.09960059566860);
    // corresponding recoil factors
    std::vector<double> rf = {6.19763968e-02, 2.11299479e-01, 3.92230225e-02, 8.37071153e-02,
       6.59541694e-01, 5.54584619e-01, 5.15891967e-01, 2.71539832e-01,
       3.62759758e+01, 2.32983838e+01, 4.49402573e+01, 4.66851721e+01,
       3.42608570e+00, 1.47585117e+00, 4.37854353e+00, 3.01875804e+00,
       2.78481909e-01, 4.44885011e-01, 5.94050168e+00, 2.39665064e+00,
       5.94050168e+00, 3.62759758e+01, 4.49402573e+01, 2.39665064e+00,
       2.32983838e+01, 4.66851721e+01, 5.94050168e+00, 3.42608570e+00,
       4.37854353e+00, 2.39665064e+00, 1.47585117e+00, 3.01875804e+00,
       4.78687810e+00, 4.78687810e+00, 2.43042367e+00, 2.43042367e+00,
       3.62759758e+01, 3.42608570e+00, 9.22485928e+00, 2.32983838e+01,
       1.47585117e+00, 7.98432109e+00, 1.25880365e+00, 1.25880365e+00,
       4.49402573e+01, 4.37854353e+00, 9.22485928e+00, 4.66851721e+01,
       3.01875804e+00, 7.98432109e+00, 4.05153711e+00, 4.05153711e+00};
    // corresponding mandelstam invariants
    std::vector<double> s =  {57620.19834447,  2481.25340357, 17526.21449119,  6151.71604714,
       11811.18214835, 15318.87014061,  4330.96211361, 30724.04784004,
         839.38541863,  3805.32768401,  4985.63476133, 10190.29636157,
        7075.50627888,  8395.56297604,  4673.64341465,  4403.16273961,
       12331.09603429,  3401.83607903,  1891.10297262,   674.73501224,
        4550.72877841,  2853.47017052,  2588.12676911,   575.9899164 ,
         228.18065838,  6129.58133862,  3001.2110501 ,  1921.77043446};
    const fdeep::tensor rfs = fdeep::tensor(fdeep::tensor_shape(52), rf);
    const fdeep::tensor sijs = fdeep::tensor(fdeep::tensor_shape(28), s);
    std::cout << fdeep::show_tensor(ps) << std::endl;
    std::cout << fdeep::show_tensor(rfs) << std::endl;
    std::cout << fdeep::show_tensor(sijs) << std::endl;
    std::cout << fdeep::show_tensor_shape(ps.shape()) << std::endl;
    std::cout << fdeep::show_tensor_shape(rfs.shape()) << std::endl;
    std::cout << fdeep::show_tensor_shape(sijs.shape()) << std::endl;

    // predict C_ijk
    const fdeep::tensors result = model.predict({ps, rfs, sijs});

    // check the  ~same (similar because of some floating point precision errors, especially for small values) as Python output:
    // [-9.72681080e-16,  6.09512573e-29,  4.30025884e-30, -8.18277535e-27,
    //         2.21777542e-30,  2.52664958e-29,  1.72666259e-19,  2.97173687e-18,
    //         2.73425857e-27,  4.19911336e-29,  5.82349766e-18,  2.50604801e-36,
    //         1.17895397e-17,  9.56274256e-19,  9.35242150e-16,  6.15001806e-13,
    //        -4.12994170e-23, -1.04530656e-19,  1.62248129e-26, -1.82816062e-22,
    //        -1.05819061e-37,  3.50984910e-26,  3.26664796e-24,  5.83502148e-32,
    //         5.30274717e-27,  1.04153566e-43,  3.77599814e-08,  1.65674051e-33,
    //         6.29806824e-34,  1.69333416e-08, -3.21294317e-31,  1.13003578e-19,
    //         1.70282352e-31, -3.38058092e-21, -7.14365996e-27, -1.47740478e-22,
    //         8.84169455e-23,  3.89961384e-21,  8.74166976e-29, -3.31835628e-28,
    //        -5.23683828e-23, -9.32712816e-29,  2.29918808e-15, -3.94319455e-22,
    //         2.62859901e-33,  2.08731443e-32,  1.45069528e-30, -1.09495631e-26,
    //        -2.83465215e-26, -8.11090184e-39,  4.46412256e-14, -3.24146873e-33,
    //        -5.07594690e-19, -5.50245877e-33,  2.58551086e-14, -1.75256339e-26,
    //         1.16856167e-22,  1.18118082e-23,  9.25681783e-18, -3.24077511e-19,
    //         5.09539046e-16, -2.04163080e-19]

    const std::vector<double> res = single_tensor_from_tensors(result).to_vector();
    for (float i: res)
        std::cout << i << ',';

    // need to multiply with D_ij,k to reproduce matrix element
}
