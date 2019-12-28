#ifndef EKF_H_
#define EKF_H_
#include <cstdlib>
#include <iostream>
#include <memory>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

class KFBase
{
private:
    /* data */

public:
    void predict(const double& stamp);
    void update(const Eigen::VectorXd &z);
    virtual void init(const double &stamp, const Eigen::VectorXd &x) {}

    Eigen::VectorXd x() const {return this->x_;}
    Eigen::MatrixXd P() const {return this->P_;}
    Eigen::MatrixXd S() const {return this->S_;}
    void setStateCoveriance(const Eigen::MatrixXd& P) {
        this->P_ = P;
    }

    void setState(const Eigen::VectorXd& x) {
        this->x_ = x;
    }

    void setCurrentTimeStamp(const double& stamp) {
        std::cout << "--------------------------------" << std::endl;
        std::cout << std::fixed << stamp << std::endl;
        std::cout << std::fixed << this->current_time_stamp_ << std::endl;
        this->dt_ = stamp - this->current_time_stamp_;
        std::cout << std::fixed << dt_ << std::endl;
        this->current_time_stamp_ = stamp;
        if (this->dt_ < 0)
            this->dt_ = 1e-4;
    }

    double likelihood() const {return this->likelihood_;}
    virtual KFBase* clone() {return new KFBase(*this);}
    // virtual void init(const Eigen::MatrixXd &P, const Eigen::MatrixXd &R, const double &dt) = 0;
protected:
    Eigen::MatrixXd F_; // 状态转移矩阵
    Eigen::MatrixXd H_; // 测量矩阵
    Eigen::MatrixXd Q_; // 系统协方差矩阵
    Eigen::MatrixXd R_; // 测量协方差矩阵
    Eigen::MatrixXd J_; // 雅可比阵
    Eigen::MatrixXd P_; // 过程协方差矩阵
    Eigen::MatrixXd S_;
    Eigen::VectorXd x_; // 状态向量
    Eigen::VectorXd z_; // 测量向量
    
    std::vector<bool> angle_mask_;
    double likelihood_;
    double dt_;
    double current_time_stamp_;
    virtual void updatePrediction() {}
    virtual void updateMeasurement() {}

    static double normalizeAngle(const double raw_angle) //象限问题
    {
        int n = 0;
        double angle = 0;
        n = raw_angle / (3.141592653 * 2);
        angle = raw_angle - (3.141592653 * 2) * n;
        if (angle > 3.141592653)
        {
            angle = angle - (3.141592653 * 2);
        }
        else if (angle <= -3.141592653)
        {
            angle = angle + (3.141592653 * 2);
        }

        return angle;
    }
    

};

class CTRV : public KFBase
{
private:
    void init(const double &stamp, const Eigen::VectorXd &x);
    CTRV();
    ~CTRV();
};


/*****************************************
    Constant Turn Rate and Acceleration Model
    Parameter:
    X: {x, y, theta, speed, yaw_rate, acc}
    Z: {x, y, theta, speed}
*****************************************/
class CTRA : public KFBase
{
private:
    void updatePrediction();
    void updateMeasurement();
public:
    void init(const double &stamp, const Eigen::VectorXd &x);
    CTRA();
    ~CTRA();
};

class CV : public KFBase {
private:
    void updatePrediction();
    void updateMeasurement();
public:
    void init(const double &stamp, const Eigen::VectorXd &x);
    CV();
    ~CV();
};

class CA : public KFBase {
private:
    void updatePrediction();
    void updateMeasurement();
public:
    void init(const double &stamp, const Eigen::VectorXd &x);
    CA();
    ~CA();
};

class CT : public KFBase {
private:
    void updatePrediction();
    void updateMeasurement();
    const double yaw_rate_;
public:
    void init(const double &stamp, const Eigen::VectorXd &x);
    CT(const double& yaw_rate);
    ~CT();
};


#endif