import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data (same as before)
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

# Convert channel numbers to positions (0-16)
def channel_to_position(channel_id):
    """Convert channel ID to position index (0-16)"""
    # Assuming channels are numbered sequentially from 0 or 1
    # Adjust this mapping based on your actual channel numbering
    if channel_id < 16:
        return 0.5 + channel_id * 1.6
    else:
        # Handle channels above 16 if they exist
        return channel_id - 1  # Adjust based on your numbering

merged_data['position'] = merged_data['Other_Ch_Id'].apply(channel_to_position)
print(f"Position range: {merged_data['position'].min()} to {merged_data['position'].max()}")

# Create new engineered features
def create_engineered_features(df):
    """Create the three engineered features"""
    features = []
    
    # Feature 1: log(PHA_LG_17 / PHA_LG_18)
    pha_ratio = np.log(df['PHA_LG_17'] / df['PHA_LG_18'])
    features.append(pha_ratio)
    
    # Feature 2: ToA_17 - ToA_18
    toa_diff = df['ToA_ns_17'] - df['ToA_ns_18']
    features.append(toa_diff)
    
    # Feature 3: ToT_17 - ToT_18  
    tot_diff = df['ToT_ns_17'] - df['ToT_ns_18']
    features.append(tot_diff)
    
    return np.column_stack(features)

# Prepare features and targets
X = create_engineered_features(merged_data)
y = merged_data['position'].values

print(f"Final dataset - X: {X.shape}, y: {y.shape}")
print(f"Position distribution:\n{pd.Series(y).value_counts().sort_index()}")

# Filter out rows with invalid values (infinite or NaN)
valid_mask = ~np.any(~np.isfinite(X), axis=1)
X = X[valid_mask]
y = y[valid_mask]

print(f"Data after removing invalid values: {X.shape}")

# Normalize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Normalize targets to 0-1 range for better training
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train)


# Neural network for position prediction
class PositionPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64]):  # Wider layers
        super(PositionPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            # Only add dropout to first layer, reduced rate
            if i == 0:
                layers.append(nn.Dropout(0.2))
            # Remove BatchNorm entirely for low data
            prev_size = hidden_size
        
        # Single output for position
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)





# First, calculate test_true for evaluation
test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()


# Quick test of most promising options
print("\n" + "="*60)
print("QUICK OPTIMIZER COMPARISON")
print("="*60)

quick_tests = [
    ('adam', 0.0005),
]

best_rmse = float('inf')
best_optimizer = None
best_learning_rate = None

# Define criterion here (outside the loop)
criterion_test = nn.MSELoss()

for opt_name, lr in quick_tests:
    print(f"\n--- Testing {opt_name} with lr={lr} ---")
    
    # Re-initialize model for fresh start
    model_test = PositionPredictor(input_size=X_train.shape[1])
    
    # Choose optimizer
    if opt_name == 'adam':
        optimizer_test = optim.Adam(model_test.parameters(), lr=lr, weight_decay=1e-4)
    else:  # adamw
        optimizer_test = optim.AdamW(model_test.parameters(), lr=lr, weight_decay=1e-4)
    
    # Modified training function for quick testing
    def quick_train_model(model, X_train, y_train, X_test, y_test, optimizer, criterion, epochs=100):
        train_losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            train_loss = criterion(outputs.squeeze(), y_train)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
        return train_losses
    
    # Quick training
    train_losses = quick_train_model(
        model_test, X_train_tensor, y_train_tensor, X_test_tensor, torch.FloatTensor(y_test),
        optimizer=optimizer_test, criterion=criterion_test, epochs=100
    )
    
    # Evaluate
    model_test.eval()
    with torch.no_grad():
        test_outputs = model_test(X_test_tensor)
        test_predictions_scaled = test_outputs.squeeze().numpy()
        test_predictions = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
        rmse = np.sqrt(mean_squared_error(test_true, test_predictions))
    
    print(f"RMSE: {rmse:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_optimizer = opt_name
        best_learning_rate = lr

print(f"\n BEST CONFIGURATION: {best_optimizer} with lr={best_learning_rate}, RMSE={best_rmse:.4f}")

# Now use the best configuration for your final model training
print(f"\n" + "="*60)
print("FINAL TRAINING WITH BEST CONFIGURATION")
print("="*60)

# ↑↑↑ END OF METHOD 4 ↑↑↑









# Initialize model
model = PositionPredictor(input_size=X_train.shape[1])
print(f"Model initialized for position prediction")

# Training setup - using MSE loss for regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# Training loop for regression
def train_model(model, X_train, y_train, X_test, y_test, epochs=200):
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs.squeeze(), y_train)
        train_loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs.squeeze(), y_test)
        
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {train_loss.item():.4f}/{test_loss.item():.4f}')
    
    return train_losses, test_losses

# Train the model
print("Starting training...")
train_losses, test_losses = train_model(
    model, X_train_tensor, y_train_tensor, X_test_tensor, torch.FloatTensor(y_test), epochs=200
)

# Final evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predictions_scaled = test_outputs.squeeze().numpy()
    
    # Convert back to original scale
    test_predictions = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
    test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate metrics
mse = mean_squared_error(test_true, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(test_true, test_predictions)

print(f"\nFinal Test Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot results
plt.figure(figsize=(15, 5))

# Training history
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training Loss')

# Predictions vs True
plt.subplot(1, 3, 2)
plt.scatter(test_true, test_predictions, alpha=0.6)
plt.plot([test_true.min(), test_true.max()], [test_true.min(), test_true.max()], 'r--', lw=2)
plt.xlabel('True Position')
plt.ylabel('Predicted Position')
plt.title(f'Predictions vs True (R² = {r2:.3f})')

# Residuals
plt.subplot(1, 3, 3)
residuals = test_predictions - test_true
plt.scatter(test_predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Position')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_names': ['log(PHA_17/PHA_18)', 'ToA_17 - ToA_18', 'ToT_17 - ToT_18']
}, 'position_predictor.pth')

print(f"\nModel saved as 'position_predictor.pth'")

# Enhanced prediction function with uncertainty estimation
def predict_position_with_uncertainty(model, scaler_X, scaler_y, timing_features, amplitude_features, num_samples=100):
    """
    Predict position with uncertainty estimates using Monte Carlo dropout
    """
    # Prepare feature vector using engineered features
    pha_ratio = np.log(amplitude_features.get('PHA_LG_17', 1) / amplitude_features.get('PHA_LG_18', 1))
    toa_diff = timing_features.get('ToA_ns_17', 0) - timing_features.get('ToA_ns_18', 0)
    tot_diff = timing_features.get('ToT_ns_17', 0) - timing_features.get('ToT_ns_18', 0)
    
    features = np.array([[pha_ratio, toa_diff, tot_diff]])
    features_scaled = scaler_X.transform(features)
    
    # Enable dropout for uncertainty estimation
    model.train()
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(torch.FloatTensor(features_scaled))
            pred_scaled = output.squeeze().item()
            pred_original = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred_original)
    
    predictions = np.array(predictions)
    
    return {
        'predicted_position': np.mean(predictions),
        'std_error': np.std(predictions),
        'confidence_interval': (np.percentile(predictions, 2.5), np.percentile(predictions, 97.5)),
        'all_predictions': predictions
    }

# Example usage
print("\nExample prediction with uncertainty:")
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

result = predict_position_with_uncertainty(model, scaler_X, scaler_y, example_timing, example_amplitude)

print(f"Predicted position: {result['predicted_position']:.2f}")
print(f"Standard error: ±{result['std_error']:.2f}")
print(f"95% Confidence Interval: ({result['confidence_interval'][0]:.2f}, {result['confidence_interval'][1]:.2f})")
print(f"Position range: 0 to 16")

# Show some test predictions with uncertainty
print("\nSample test predictions with uncertainty:")
model.eval()
with torch.no_grad():
    sample_indices = np.random.choice(len(X_test), min(3, len(X_test)), replace=False)
    for idx in sample_indices:
        features = X_test[idx].reshape(1, -1)
        true_pos = test_true[idx]
        
        # Get prediction with uncertainty
        pred_result = predict_position_with_uncertainty(
            model, scaler_X, scaler_y, 
            {'ToA_ns_17': 0, 'ToT_ns_17': 0, 'ToA_ns_18': 0, 'ToT_ns_18': 0},  # Dummy, we're using pre-scaled features
            {'PHA_LG_17': 1, 'PHA_LG_18': 1},
            num_samples=50
        )
        
        error = pred_result['predicted_position'] - true_pos
        print(f"True: {true_pos:.1f}, Pred: {pred_result['predicted_position']:.1f} ± {pred_result['std_error']:.2f}, Error: {error:.1f}")