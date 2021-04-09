# Privacy in FL systems
Experiments on Privacy of Federated Learning systems
We are going to implement Softmax Regression (SR)
for multi-class classification of provided datasets. 
In order to optimize parameters we 
use Stochastic Gradient Descent (SGD)
approach with batches of size
100 samples and with 20 iterations (i.e. epochs) which 
is enough to smoothly run the system and see the differences. As suggested by SR, we use Cross-Entropy Loss as our loss function.
We will implement and compare different scenarios to get an idea 
of how each component affect the accuracy.

## Dataset
[MNIST](http://yann.lecun.com/exdb/mnist/) dataset of handwritten digits. It contains 60K training samples and 10K testing samples.

## Scenario 1
For the first scenario we just implement the standard SR algorithm and evaluate it on a centralized server to see how it works with the provided data. We randomly split the dataset into three different partitions 50K for training, 10K for validation and 10K for testing.

## Scenario 2
The researches showed that a FL system will reduce the accuracy of the model compared to its equivalent centralized model. Based on this idea, as the second scenario, we first simulate a FL system on a centralized server. For this simulation we considered 10 different clients each with their own dataset and without any conflict with others. Then, we implement the same SR model for each client and server. The aggregation at the server will be done by taking average of clients' model parameters (i.e. all clients have the same weight of 0.1). We randomly assigned 20K samples to the server (i.e. 14K for training, 2K for validation and 4K for testing) and 5K to each client (i.e. 4K for training and 1K for validation)

## Scenario 3
In this scenario we try to add Gaussian noise to the model parameters of the implemented FL system
