#include <iostream>
#include <vector>
#include <omp.h>

// Define the linear regression model
class LinearRegression {
public:
    LinearRegression(double learning_rate) : learning_rate(learning_rate) {}

    void train(std::vector<double>& X, std::vector<double>& y, int num_epochs) {
        int m = X.size();
        double b = 0, w = 0;
        #pragma omp parallel for shared(b, w)
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            double b_grad = 0, w_grad = 0;
            for (int i = 0; i < m; i++) {
                double y_pred = b + w * X[i];
                double error = y_pred - y[i];
                b_grad += error;
                w_grad += error * X[i];
            }
            b_grad /= m;
            w_grad /= m;
            #pragma omp atomic
            b -= learning_rate * b_grad;
            #pragma omp atomic
            w -= learning_rate * w_grad;
        }
        this->b = b;
        this->w = w;
    }

    double predict(double x) {
        return b + w * x;
    }

private:
    double b, w;
};
