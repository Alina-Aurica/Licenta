# optimizes the parameters based on gradients
import torch.optim as optim

from some_random_model.CNN import *

# loss function - measure network's performance
#               - it measures the difference between two probability distributions: the actual labels and the predictions made by the network
criterion = nn.CrossEntropyLoss()
# optimizer - update the network weights in the direction that reduces the loss
#           - initializes a stochastic gradient descent (SGD) optimizer
# hyperparameters:
#           - net.parameters() - specifies witch parameters should be updated => all params here
#           - lr - learning rate
#           - momentum - helps accelerate SGD in the relevant direction + dampens oscillations
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.7, alpha=0.70)