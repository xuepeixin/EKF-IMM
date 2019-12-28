#include <cstdlib>
#include <iostream>
#include <memory>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "EKF.h"
#include "IMM.h"
int main() {
    // CTRA ctra;
    // Eigen::VectorXd x;
    // Eigen::MatrixXd P;
    // P.resize(6,6);
    // x.resize(6);
    // P <<   0.00361139,  -0.00161339,  0.00162952 , -0.00302107 ,  0.00358926 , -0.00173016,
    //       -0.00161339,   0.00611165, -0.00334304 , -0.00147591 , -0.00745283 , -0.000846224,
    //        0.00162952,  -0.00334304,  0.00381487 , -3.98198e-05,  0.0115538  , -1.89119e-05,
    //       -0.00302107,  -0.00147591, -3.98198e-05,  0.0102369  , -8.6498e-05 ,  0.00714289,
    //        0.00358926,  -0.00745283,  0.0115538  , -8.6498e-05 ,  0.0640082  , -4.55512e-05,
    //       -0.00173016,  -0.00084622, -1.89119e-05,  0.00714289 , -4.55512e-05,  0.0101376;
    // x << -10846.3, -2582.08, -2.69819, 8.59548, -0.064526, -0.0291316;
    // std::vector<bool> mask = {false, false, true, false};
    // ctra.init(0.04, x);
    // ctra.setStateCoveriance(P);
    // std::cout << "e" << std::endl;
    // std::vector<Eigen::Matrix<double, 6, 1>> x_list;
    // std::vector<Eigen::Matrix<double, 6, 1>> covariance_list;
    // for (int i = 0; i < 50; i++) {
    //     x_list.push_back(ctra.x());
    //     Eigen::Matrix<double, 6, 1> covariance;
    //     covariance << ctra.P()(0, 0), ctra.P()(1, 1), ctra.P()(2, 2), 
    //                     ctra.P()(3, 3), ctra.P()(4, 4), ctra.P()(5, 5);
    //     covariance_list.push_back(covariance);

    //     ctra.Predict();
    //     // std::cout << "---------------------------------" << std::endl;
    //     // std::cout << ctra.x() << std::endl;
    //     // std::cout << ctra.P() << std::endl;
    // }
    // std::cout << "[";
    // for (auto it:x_list) {
    //     std::cout << "[" << it(0) << ", " 
    //                      << it(1) << ", "
    //                      << it(2) << ", "
    //                      << it(3) << ", "
    //                      << it(4) << ", "
    //                      << it(5) << "], ";
    // }
    // std::cout << "]\n";

    // std::cout << "[";
    // for (auto it:covariance_list) {
    //     std::cout << "[" << it(0) << ", " 
    //                      << it(1) << ", "
    //                      << it(2) << ", "
    //                      << it(3) << ", "
    //                      << it(4) << ", "
    //                      << it(5) << "], ";
    // }
    // std::cout << "]\n";


    IMM imm;
    std::shared_ptr<KFBase> cv = std::shared_ptr<KFBase>(new CV());
    std::shared_ptr<KFBase> ca = std::shared_ptr<KFBase>(new CA());
    std::shared_ptr<KFBase> ct0 = std::shared_ptr<KFBase>(new CT(0.01));
    std::shared_ptr<KFBase> ct1 = std::shared_ptr<KFBase>(new CT(-0.01));
    imm.addModel(cv);
    imm.addModel(ca);
    imm.addModel(ct0);
    imm.addModel(ct1);
    Eigen::MatrixXd transfer_prob = Eigen::MatrixXd::Zero(4,4);
    transfer_prob << 0.8, 0.1, 0.05, 0.05,
                     0.1, 0.8, 0.05, 0.05,
                     0.1, 0.1, 0.75, 0.05,
                     0.1, 0.1, 0.05, 0.75;
    Eigen::VectorXd model_prob = Eigen::VectorXd::Zero(4);
    model_prob << 0.25, 0.25, 0.25, 0.25;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(6);
    imm.init(transfer_prob, model_prob, x, 0.025);
                     
    return 0;
}