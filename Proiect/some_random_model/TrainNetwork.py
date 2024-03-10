from some_random_model.CNN import net
from LoadImages import *
from LossFunction import optimizer, criterion
from ValidationNetwork import *
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

# training loop
# loop over the dataset multiple times (2 epochs)
# input data -> forward pass => calculate the loss + backpropagation
#                            -> update the weights with optimizer
def train_function():
    # net.train()
    for epoch in range(50):
        # init a variable to accumulate the loss for 2000 mini-batches
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        # iterates over the train data loader
        # enumerate(...) - used to keep track of the number of mini-batches processed
        pbar = tqdm(enumerate(train_loader, 0),
                    unit='image',
                    total=len(train_loader),
                    smoothing=0)

        for i, data in pbar:
            # get the inputs
            # data - a list of [inputs, labels]
            inputs, labels = data
            # clean old gradients from the last step
            optimizer.zero_grad()
            # forward pass - passes the input data through the network
            outputs = net(inputs)
            # calculates the loss
            loss = criterion(outputs, labels) # Cross Entropy (pred, target)
            # backward pass - computes the gradient of the loss with respect to all the network's parameters,
            #                 requires_grad = True
            loss.backward()
            # updates the parameters
            optimizer.step()

            # print statistics - print loss + accuracy
            # HERE SOME METRICS LIKE ACCURACY, RECALL, PRECISION, ERROR RATES
            # Predictions
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            running_samples += labels.size(0)
            running_loss += loss.item()
            accuracy = running_correct / running_samples * 100
            pbar.set_description('Train [ E {}, L {:.4f}, A {:.4f}]'.format(epoch, float(running_loss) / (i + 1), accuracy))

        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        # print(f'{epoch + 1} loss: {running_loss / 1000:.3f} accuracy: {accuracy:.2f}%')
        running_loss = 0.0

        # validation phase
        validate_model(net, validationLoader=val_loader, criterion=criterion)

log_dir = "/tensorboard/test5Adam"
writer = SummaryWriter(log_dir=log_dir)

train_function()
print('Finished Training')

# save the trained model
PATH = '../grocery_net.pth'
torch.save(net.state_dict(), PATH)

# ACEST LOOP ESTE ECHIVALENT CU net.train()?