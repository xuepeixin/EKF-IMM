#ifndef IMM_H_
#define IMM_H_
#include <cstdlib>
#include <iostream>
#include <memory>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "EKF.h"


class IMM
{
private:
    std::vector<std::shared_ptr<KFBase>> models_;
    Eigen::MatrixXd transfer_prob_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd X_;
    Eigen::VectorXd c_;
    Eigen::VectorXd model_prob_;
    Eigen::VectorXd x_;
    size_t model_num_;
    size_t state_num_;
    double current_time_stamp_;
public:
    void addModel(const std::shared_ptr<KFBase>& model);
    void init(
        const Eigen::MatrixXd& transfer_prob, 
        const Eigen::VectorXd& model_prob, 
        const Eigen::VectorXd& x,
        const double& dt);
    void stateInteraction();
    void updateState(const double& stamp, const Eigen::VectorXd* z = nullptr);
    void updateModelProb();
    void estimateFusion();
    void updateOnce(const double& stamp, const Eigen::VectorXd* z = nullptr);
    Eigen::VectorXd x() const {return x_;}
    IMM(const IMM& imm) {
        transfer_prob_ = imm.transfer_prob_;
        P_ = imm.P_;
        X_ = imm.X_;
        c_ = imm.c_;
        model_prob_ = imm.model_prob_;
        x_ = imm.x_;
        model_num_ = imm.model_num_;
        state_num_ = imm.state_num_;
        for (size_t i = 0; i < imm.models_.size(); i++) {
            std::shared_ptr<KFBase> m = std::shared_ptr<KFBase>(imm.models_[i]->clone());
        }
    }
    double stamp() const {return current_time_stamp_;}
    IMM* clone() {return new IMM(*this);}

    IMM();
    ~IMM();
};



#endif