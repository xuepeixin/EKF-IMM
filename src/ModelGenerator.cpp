#include "ModelGenerator.h"
#include "EKF.h"
#include "IMM.h"

ModelGenerator::ModelGenerator(/* args */)
{
}

ModelGenerator::~ModelGenerator()
{
}


std::shared_ptr<IMM> ModelGenerator::generateIMMModel(const double &stamp, const Eigen::VectorXd &x) {
    std::shared_ptr<IMM> imm_ptr = std::shared_ptr<IMM>(new IMM());

    auto cv  = generateCVModel(stamp, x);
    auto ca  = generateCAModel(stamp, x);
    auto ct0 = generateCTModel(stamp, x, 0.1);
    auto ct1 = generateCTModel(stamp, x, -0.1);
    imm_ptr->addModel(ca);
    imm_ptr->addModel(cv);
    imm_ptr->addModel(ct0);
    imm_ptr->addModel(ct1);

    // Eigen::MatrixXd transfer_prob = Eigen::MatrixXd::Zero(2,2);
    // transfer_prob << 0.9, 0.1,
    //                 0.2, 0.8;
    // Eigen::VectorXd model_prob = Eigen::VectorXd::Zero(2);
    // model_prob << 0.5, 0.5;


    Eigen::MatrixXd transfer_prob = Eigen::MatrixXd::Zero(4,4);
    transfer_prob << 0.8, 0.1, 0.05, 0.05,
                    0.2, 0.7, 0.05, 0.05,
                    0.1, 0.1, 0.75, 0.05,
                    0.1, 0.1, 0.05, 0.75;
    Eigen::VectorXd model_prob = Eigen::VectorXd::Zero(4);
    model_prob << 0.3, 0.2, 0.25, 0.25;

    imm_ptr->init(transfer_prob, model_prob, x, stamp);

    return imm_ptr;
}

std::shared_ptr<KFBase> ModelGenerator::generateCTRAModel(const double &stamp, const Eigen::VectorXd &x) {
    std::shared_ptr<KFBase> ctra_ptr = std::shared_ptr<KFBase>(new CTRA());
    ctra_ptr->init(stamp, x);
    return ctra_ptr;
}

std::shared_ptr<KFBase> ModelGenerator::generateCVModel(const double &stamp, const Eigen::VectorXd &x) {
    std::shared_ptr<KFBase> cv_ptr = std::shared_ptr<KFBase>(new CV());
    cv_ptr->init(stamp, x);
    return cv_ptr;
}

std::shared_ptr<KFBase> ModelGenerator::generateCAModel(const double &stamp, const Eigen::VectorXd &x) {
    std::shared_ptr<KFBase> ca_ptr = std::shared_ptr<KFBase>(new CA());
    ca_ptr->init(stamp, x);
    return ca_ptr;
}

std::shared_ptr<KFBase> ModelGenerator::generateCTModel(const double &stamp, const Eigen::VectorXd &x, const double& yaw_rate) {
    std::shared_ptr<KFBase> ct_ptr = std::shared_ptr<KFBase>(new CT(yaw_rate));
    ct_ptr->init(stamp, x);
    return ct_ptr;
}