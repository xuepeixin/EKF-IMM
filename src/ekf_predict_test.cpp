#include <cstdlib>
#include <iostream>
#include <memory>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "EKF.h"
#include "IMM.h"
int main() {

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