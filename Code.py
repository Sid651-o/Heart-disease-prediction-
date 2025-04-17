import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Load the dataset
filepath = r'D:\project2\heart.csv'
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(filepath, names=columns, na_values='?', skiprows=1)

# Data Preprocessing
data = data.dropna()
data['target'] = data['target'].astype(int)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values  # Keep y as class indices (0 or 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)  # Use long for class indices

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the Fully Connected Model with Average Pooling
class HeartDiseaseFCNWithAvgPooling(nn.Module):
    def __init__(self):
        super(HeartDiseaseFCNWithAvgPooling, self).__init__()
        self.fc1 = nn.Linear(13, 64)  # Input layer (13 features) to hidden layer (64 units)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)  # Average pooling layer with kernel_size 2 and stride 2
        self.fc2 = nn.Linear(32, 128)  # Hidden layer (32 units) to next hidden layer (128 units)
        self.fc3 = nn.Linear(128, 2)   # Output layer (2 units for classification)
        self.sigmoid = nn.Sigmoid()    # Sigmoid activation function
        self.softplus = nn.Softplus()  # Softplus activation function

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))  # Apply Sigmoid activation after first fully connected layer
        x = x.unsqueeze(1)  # Add a dimension to apply pooling
        x = self.pool1(x)  # Apply average pooling
        x = x.squeeze(1)  # Remove the added dimension
        x = self.sigmoid(self.fc2(x))  # Apply Sigmoid activation after second fully connected layer
        x = self.fc3(x)  # Output layer
        x = self.softplus(x)  # Apply Softplus activation after output layer
        return x

model = HeartDiseaseFCNWithAvgPooling()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss combines Softmax and NLLLoss
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9, eps=1e-8)

# Training the model
n_epochs = 250
for epoch in range(n_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    output = model(X_test)
    _, predicted = torch.max(output, 1)  # Get the predicted class index

# Print classification report
print("Classification Report:")
print(classification_report(y_test, predicted))

# Save the model and scaler
torch.save(model.state_dict(), 'model/heart_disease_fcn_with_avg_pooling.pth')
joblib.dump(scaler, 'model/scaler.pkl')

# Function to predict heart disease for new data
def predict_heart_disease_from_csv(input_csv):
    new_data = pd.read_csv(input_csv, names=columns, na_values='?', skiprows=1)
    new_data = new_data.dropna()
    new_data['target'] = new_data['target'].astype(int)
    X_new = new_data.iloc[:, :-1].values
    X_new = scaler.transform(X_new)
    
    X_new = torch.tensor(X_new, dtype=torch.float32)
    model = HeartDiseaseFCNWithAvgPooling()
    model.load_state_dict(torch.load('model/heart_disease_fcn_with_avg_pooling.pth'))
    model.eval()
    with torch.no_grad():
        output = model(X_new)
        _, predictions = torch.max(output, 1)  # Get the predicted class index
    results = ["Heart Disease Detected" if pred == 1 else "No Heart Disease" for pred in predictions]
    return results

input_csv = r'D:\project2\heart.csv'
prediction_results = predict_heart_disease_from_csv(input_csv)
print("Prediction results for the new data from CSV:")
print(prediction_results)