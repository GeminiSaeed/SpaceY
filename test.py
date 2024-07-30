import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from IPython.display import display

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize the model and move it to the device
model = SimpleNet().to(device)

# Visualize the model
x = torch.randn(1, 10).to(device)
dot = make_dot(model(x), params=dict(model.named_parameters()))

# Customize the graph
dot.attr(rankdir='LR', size='8,8')
dot.attr('node', shape='box', style='filled', color='lightblue')
dot.attr('edge', color='darkgreen')

# Save and display the graph
dot.render("model_architecture", format="png", cleanup=True)
display(dot)

# Create random input and target data
x = torch.randn(100, 10).to(device)
y = torch.randn(100, 1).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

print("Training complete!")

# Test the model
with torch.no_grad():
    test_input = torch.randn(1, 10).to(device)
    prediction = model(test_input)
    print(f"Test prediction: {prediction.item():.4f}")