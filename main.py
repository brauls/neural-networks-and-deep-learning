import src.mnist_loader as mnist_loader
import src.network as network
import src.network2 as network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Network 1
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# Network 2
net = network2.Network([784, 30, 10])
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
