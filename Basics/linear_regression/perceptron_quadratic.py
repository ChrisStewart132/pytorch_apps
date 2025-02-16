import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
# TensorBoard setup
writer = SummaryWriter()  # Create a SummaryWriter object
# open a seperate terminal in this dir and run tensorboard with cmd below, than goto http://localhost:6006/
# tensorboard --logdir=runs

"""
given the input x and output y, use linear regression (solving x in A.x = b) to find a quadratic equation that best fits the dataset

 y = a.x**2 + b.x + c
from linear algebra 1(height) x 3(width).1 x 3 = 3 x 1
    A.x = b
        [a11 a21 a31][x1] = [y1]
                     [x2]   
                     [x3]   
    or
        [x**2, x, 1][a] = [y1]
                    [b]   [y2]
                    [c]   [y3]

In a single perceptron/node neutral network,
    x = input vector = [x1,x2]
    y = perceptron output = [y1]
    w = perceptron weights = [w1, w2, bias]
or
    x = [x**2, x]
    y = [y]
    w = [a, b, c]
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Default device: {device}")

# synthetic data that happens to perfectly fit a quadratic
# this could be any function(x), e.g. sin(x), m.x+c, a.x**3+b.x**2+c.x+d, random(x)
# this particulat script will attempt to predict a quadratic (from the perceptrons weights+bias) to best fit the dataset...
def quadratic(x):
    return 3*x**2 + 2*x + 7

# manually take our x from the dataset and add x**2 to fit to a quadratic (could also add x**3 for a polynomial and so on... or some other type of equation fn(x))
xs = [[x**2, x] for x in range(-100000, 100000)]
# normalize xs
xs1,xs2 = zip(*xs)

largest1, largest2 = max(xs1), max(xs2)

xs = [[x1/largest1, x2/largest2] for x1,x2 in xs]

ys = [[quadratic(x[1])] for x in xs]

X = torch.tensor(xs, dtype=torch.float32, device=device)
Y = torch.tensor(ys, dtype=torch.float32, device=device)

n_samples, n_features = X.shape
print(f'n_samples = {n_samples}, n_features = {n_features}')

# 0) create a test sample
X_test = torch.tensor([25.0, 5.0], dtype=torch.float32, device=device)


# 1) Design Model, the model has to implement the forward pass!

# Here we could simply use a built-in model from PyTorch
# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # define different layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


input_size, output_size = n_features, 1



model = LinearRegression(input_size, output_size)
model.to(device)

print(f'Prediction before training: f({X_test[1].item()}) = {model(X_test).item()}')

# 2) Define loss and optimizer
learning_rate = 0.001# make sure the learning rate isn't too high for complex predictions
n_epochs = 20000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(n_epochs):
    # predict = forward pass with our model
    y_predicted = model(X)

    # loss
    l = loss(y_predicted, Y)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()


    # TensorBoard logging
    writer.add_scalar('Loss/train', l.item(), epoch)  # Log the loss
    w, b = model.parameters()  # unpack parameters
    writer.add_scalar('Weights/w1', w[0][0].item(), epoch)  # Log w1
    writer.add_scalar('Weights/w2', w[0][1].item(), epoch)  # Log w2
    writer.add_scalar('Bias/b', b.item(), epoch)  # Log b


    if (epoch+1) % (n_epochs//10) == 0:
        w, b = model.parameters() # unpack parameters
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l.item())


print(f'Prediction after training: f({X_test[1].item()}) = {model(X_test).item()}')
# Calculate the actual quadratic value for comparison
actual_value = quadratic(X_test[1].item())
print(f"Actual value: f({X_test[1].item()}) = {actual_value}")


print("Note that w1,w2,b1 should predict a,b in a.x**2 + b.x + c = y")
print("Trained Weights:")
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name}: {param.data}")  # Print the weight tensor
    if 'bias' in name:
        print(f"{name}: {param.data}")  # Print the bias tensor    


writer.close()  # Close the SummaryWriter when finished
