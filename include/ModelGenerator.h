#ifndef MODEL_GENERATOR_H_
#define MODEL_GENERATOR_H_
#include "IMM.h"
#include "EKF.h"

class ModelGenerator
{
private:
    /* data */
public:
    ModelGenerator(/* args */);
    ~ModelGenerator();
    static std::shared_ptr<IMM> generateIMMModel(const double &stamp, const Eigen::VectorXd &x);
    static std::shared_ptr<KFBase> generateCTRAModel(const double &stamp, const Eigen::VectorXd &x);
    static std::shared_ptr<KFBase> generateCVModel(const double &stamp, const Eigen::VectorXd &x);
    static std::shared_ptr<KFBase> generateCAModel(const double &stamp, const Eigen::VectorXd &x);
    static std::shared_ptr<KFBase> generateCTModel(const double &stamp, const Eigen::VectorXd &x, const double& yaw_rate);


};




#endif