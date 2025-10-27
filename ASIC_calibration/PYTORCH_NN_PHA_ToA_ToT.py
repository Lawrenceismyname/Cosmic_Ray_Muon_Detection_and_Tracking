import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the data
timing_data = pd.read_csv('processed_timing_data.csv')
amplitude_data = pd.read_csv('processed_validated_positions_PHA.csv')

print(f"Timing data shape: {timing_data.shape}")
print(f"Amplitude data shape: {amplitude_data.shape}")

# First, let's find the target positions (Other_Ch_Id)
def find_other_channels(df):
    """Find the channel that is NOT 16, 17, or 18 for each event"""
    other_channels = []
    for idx in df['Index'].unique():
        event_data = df[df['Index'] == idx]
        # Find channels that are NOT 16, 17, or 18
        other_ch = event_data[~event_data['CH_Id'].isin([16, 17, 18])]
        if len(other_ch) > 0:
            other_channels.append({
                'Index': idx,
                'Other_Ch_Id': other_ch['CH_Id'].iloc[0]
            })
    return pd.DataFrame(other_channels)

# Find target positions from timing data
target_positions_timing = find_other_channels(timing_data)
print(f"Found {len(target_positions_timing)} events with other channels in timing data")

# Find target positions from amplitude data  
target_positions_amplitude = find_other_channels(amplitude_data)
print(f"Found {len(target_positions_amplitude)} events with other channels in amplitude data")

# Use timing data as primary since it has more events
target_positions = target_positions_timing

# Preprocess timing data - group by index
timing_pivot = timing_data.pivot_table(
    index='Index', 
    columns='CH_Id', 
    values=['ToA_ns', 'ToT_ns'],
    aggfunc='first'
)
timing_pivot.columns = [f'{col[0]}_{col[1]}' for col in timing_pivot.columns]
timing_pivot = timing_pivot.reset_index()

# Preprocess amplitude data - group by index
amplitude_pivot = amplitude_data.pivot_table(
    index='Index',
    columns='CH_Id',
    values='PHA_LG',
    aggfunc='first'
)
amplitude_pivot.columns = [f'PHA_LG_{col}' for col in amplitude_pivot.columns]
amplitude_pivot = amplitude_pivot.reset_index()

# Merge all datasets
merged_data = pd.merge(timing_pivot, amplitude_pivot, on='Index', how='inner')
merged_data = pd.merge(merged_data, target_positions, on='Index', how='inner')

print(f"Merged data shape: {merged_data.shape}")
print(f"Target channel distribution:\n{merged_data['Other_Ch_Id'].value_counts().sort_index()}")

# Prepare features (X) - using channels 17 and 18 only
feature_columns = []
for channel in [17, 18]:
    feature_columns.extend([
        f'ToA_ns_{channel}',
        f'ToT_ns_{channel}', 
        f'PHA_LG_{channel}'
    ])

print(f"Using features: {feature_columns}")

# Filter out rows with missing values
filtered_data = merged_data[feature_columns + ['Other_Ch_Id']].dropna()
print(f"Data after removing NaN: {filtered_data.shape}")

# Prepare features (X) and targets (y)
X = filtered_data[feature_columns].values
y = filtered_data['Other_Ch_Id'].values

print(f"Final dataset - X: {X.shape}, y: {y.shape}")
print(f"Unique target channels: {np.unique(y)}")

# Encode target channels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"Encoded classes: {label_encoder.classes_}")
print(f"Number of classes: {num_classes}")

# Check class distribution and remove classes with too few samples
class_counts = pd.Series(y).value_counts()
print(f"Class distribution:\n{class_counts}")

# Remove classes with only 1 sample (optional - or we can use regular split)
min_samples_per_class = 2
valid_classes = class_counts[class_counts >= min_samples_per_class].index
mask = np.isin(y, valid_classes)

if np.sum(mask) < len(y):
    print(f"Removing {len(y) - np.sum(mask)} samples from classes with < {min_samples_per_class} samples")
    X = X[mask]
    y = y[mask]
    y_encoded = label_encoder.transform(y)  # Re-transform after filtering

print(f"Data after filtering rare classes: {X.shape}")

# Normalize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Split data - REMOVED STRATIFY due to small class sizes
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
    # Removed: stratify=y_encoded
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Neural network for channel classification
class ChannelClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32], num_classes=num_classes):
        super(ChannelClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# Initialize model
model = ChannelClassifier(input_size=X_train.shape[1])
print(f"Model initialized for {num_classes}-class classification")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# Training loop
def train_model(model, X_train, y_train, X_test, y_test, epochs=200):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Calculate training accuracy
        _, train_preds = torch.max(outputs, 1)
        train_acc = accuracy_score(y_train.numpy(), train_preds.numpy())
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            _, test_preds = torch.max(test_outputs, 1)
            test_acc = accuracy_score(y_test.numpy(), test_preds.numpy())
        
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {train_loss.item():.4f}/{test_loss.item():.4f}, '
                  f'Acc: {train_acc:.3f}/{test_acc:.3f}')
    
    return train_losses, test_losses, train_accuracies, test_accuracies

# Train the model
print("Starting training...")
train_losses, test_losses, train_accuracies, test_accuracies = train_model(
    model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs=200
)

# Final evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_preds = torch.max(test_outputs, 1)
    
    test_accuracy = accuracy_score(y_test, test_preds.numpy())
    test_predictions = label_encoder.inverse_transform(test_preds.numpy())
    test_true = label_encoder.inverse_transform(y_test)

print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(test_true, test_predictions))

# Plot results
plt.figure(figsize=(15, 5))

# Training history
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy')

# Feature importance (using first layer weights)
plt.subplot(1, 3, 3)
first_layer_weights = model.network[0].weight.detach().numpy()
feature_importance = np.mean(np.abs(first_layer_weights), axis=0)
feature_names = ['ToA_17', 'ToT_17', 'PHA_17', 'ToA_18', 'ToT_18', 'PHA_18']

plt.bar(feature_names, feature_importance)
plt.xticks(rotation=45)
plt.ylabel('Average Weight Magnitude')
plt.title('Feature Importance')

plt.tight_layout()
plt.show()

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'label_encoder': label_encoder,
    'feature_columns': feature_columns
}, 'channel_classifier.pth')

print(f"\nModel saved as 'channel_classifier.pth'")

# Prediction function
def predict_channel(model, scaler_X, label_encoder, timing_features, amplitude_features):
    """
    Predict which channel fired based on measurements from channels 17 and 18
    """
    # Prepare feature vector
    features = []
    for channel in [17, 18]:
        features.extend([
            timing_features.get(f'ToA_ns_{channel}', 0),
            timing_features.get(f'ToT_ns_{channel}', 0),
            amplitude_features.get(f'PHA_LG_{channel}', 0)
        ])
    
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler_X.transform(features)
    
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(features_scaled))
        _, predicted = torch.max(outputs, 1)
        predicted_channel = label_encoder.inverse_transform(predicted.numpy())
    
    return predicted_channel[0]

# Example usage
print("\nExample prediction:")
example_timing = {
    'ToA_ns_17': 220.0,
    'ToT_ns_17': 30.0,
    'ToA_ns_18': 225.0, 
    'ToT_ns_18': 25.0
}

example_amplitude = {
    'PHA_LG_17': 4000,
    'PHA_LG_18': 2000
}

predicted_channel = predict_channel(model, scaler_X, label_encoder, example_timing, example_amplitude)
print(f"Predicted channel: {predicted_channel}")

# Show some test predictions
print("\nSample test predictions:")
model.eval()
with torch.no_grad():
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    for idx in sample_indices:
        features = X_test[idx].reshape(1, -1)
        outputs = model(torch.FloatTensor(features))
        _, predicted = torch.max(outputs, 1)
        predicted_ch = label_encoder.inverse_transform(predicted.numpy())[0]
        true_ch = label_encoder.inverse_transform([y_test[idx]])[0]
        print(f"True: {true_ch}, Predicted: {predicted_ch} {'✓' if predicted_ch == true_ch else '✗'}")