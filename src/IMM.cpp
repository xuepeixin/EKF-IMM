#include "IMM.h"



IMM::IMM():model_num_(0)               
{
}

IMM::~IMM()
{
}



void IMM::addModel(const std::shared_ptr<KFBase>& model) {
    this->models_.push_back(model);
    this->model_num_++;
}

void IMM::init (const Eigen::MatrixXd& transfer_prob, 
                const Eigen::VectorXd& model_prob, 
                const Eigen::VectorXd& x,
                const double& dt) {
    if (this->model_num_ == 0) {
        std::cerr << "[error] No valid model." << std::endl;
        exit(1);
    }

    if (transfer_prob.cols() != this->model_num_ || transfer_prob.rows() != this->model_num_)
    {
        std::cerr << "[error] Dimension of transfer probability matrix is not equal to number of models." << std::endl;
         exit(1);
    }
    if (model_prob.size() != this->model_num_)
    {       
        std::cerr << "[error] Dimension of model probability vector is not equal to number of models." << std::endl;
         exit(1);
    }
    this->state_num_ = x.size();
    this->X_.resize(this->state_num_, this->model_num_);

    for (size_t i = 0; i < model_num_; i++) {
        this->X_.col(i) = this->models_[i]->x();
    }

    this->transfer_prob_ = transfer_prob;
    this->model_prob_ = model_prob;
    this->x_ = x;
    this->c_.resize(this->model_num_);
}

void IMM::stateInteraction() {
    this->c_ = Eigen::VectorXd::Zero(this->model_num_);
    // c(j) = sum_i(transfer_prob(i,j)*model_prob(i))
    for (size_t j = 0; j < this->model_num_; j++) {
        for (size_t i = 0; i < this->model_num_; i++) {
            this->c_(j) += this->transfer_prob_(i, j) * this->model_prob_(i);
        }
    }

    // u(i,j) = 1 / c(j) * transfer_prob(i,j) * model_prob(i)
    // X(j) = sum_i(X(i) * U(i, j))

    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(this->model_num_, this->model_num_);
    for (size_t i = 0; i < model_num_; i++) {
        this->X_.col(i) = this->models_[i]->x();
    }
    Eigen::MatrixXd X = this->X_;
    this->X_.fill(0);
    for (size_t j = 0; j < this->model_num_; j++) {
        for (size_t i = 0; i < this->model_num_; i++) {
            U(i, j) = 1.0 / this->c_(j) * this->transfer_prob_(i, j) * this->model_prob_(i);
            this->X_.col(j) += X.col(i) * U(i, j);
        }
    } 


    for (size_t i = 0; i < this->model_num_; i++) {
        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(this->state_num_, this->state_num_);
        for (size_t j = 0; j < this->model_num_; j++) {
            Eigen::VectorXd s = X.col(i) - this->X_.col(j);
            P += U(i,j) * (this->models_[i]->P() + s * s.transpose());
        }
        this->models_[i]->setStateCoveriance(P);
        this->models_[i]->setState(this->X_.col(i));
    }
}

void IMM::updateState(const double& stamp, const Eigen::VectorXd* z) {
    for (size_t i = 0; i < this->model_num_; i++) {
        this->models_[i]->predict(stamp);
        if (nullptr != z) {
            this->models_[i]->update(*z);
        }
    }
}

void IMM::updateModelProb() {
    double c_sum = 0;
    for (size_t i = 0; i < this->model_num_; i++) {
        c_sum += this->models_[i]->likelihood() * this->c_(i);
    }
    
    // std::cout << "model_prob: " << std::endl;
    for (size_t i = 0; i < this->model_num_; i++) {
        this->model_prob_(i) = 1 / c_sum * this->models_[i]->likelihood() * this->c_(i);
        // std::cout << this->model_prob_(i) << "\t";
    }
    // std::cout << std::endl;
}

void IMM::estimateFusion() {
    this->x_ = this->X_ * this->model_prob_;

    for (size_t i = 0; i < this->model_num_; i++) {
        Eigen::MatrixXd v = this->X_.col(i) - this->x_;      
        this->P_ = this->models_[i]->P() + v * v.transpose() * this->model_prob_[i];
    }
}


