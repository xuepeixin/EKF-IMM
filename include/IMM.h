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
    Eigen::VectorXd x() const {return x_;}
    
    IMM();
    ~IMM();
};



#endif