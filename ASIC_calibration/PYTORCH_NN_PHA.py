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

amplitude_data = pd.read_csv('processed_validated_positions_PHA.csv')

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


# Find target positions from amplitude data  
target_positions_amplitude = find_other_channels(amplitude_data)
print(f"Found {len(target_positions_amplitude)} events with other channels in amplitude data")

# Use timing data as primary since it has more events
target_positions = target_positions_amplitude

# Preprocess amplitude data - group by index
amplitude_pivot = amplitude_data.pivot_table(
    index='Index',
    columns='CH_Id',
    values='PHA_LG',
    aggfunc='first'
)
amplitude_pivot.columns = [f'PHA_LG_{col}' for col in amplitude_pivot.columns]
amplitude_pivot = amplitude_pivot.reset_index()


merged_data = pd.merge(amplitude_pivot, target_positions, on='Index', how='inner')

print(f"Merged data shape: {merged_data.shape}")
print(f"Target channel distribution:\n{merged_data['Other_Ch_Id'].value_counts().sort_index()}")

# Convert channel numbers to positions (0-25)
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
    
    # # Feature 1: log(PHA_LG_17 / PHA_LG_18)
    # pha_ratio = np.log(df['PHA_LG_17'] / df['PHA_LG_18'])
    # features.append(pha_ratio)

    # pha_ratio = (df['PHA_LG_17'] - df['PHA_LG_18']) / (df['PHA_LG_17'] + df['PHA_LG_18'])
    # features.append(pha_ratio)

    pha_ratio = df['PHA_LG_17']
    features.append(pha_ratio)

    pha_ratio = df['PHA_LG_18']
    features.append(pha_ratio)
    
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


from sklearn.model_selection import train_test_split
import pandas as pd

def stratified_split_positions(X, y, val_size=0.2, test_size=0.2, min_samples_per_position=2):
    """
    Ensure each position has at least min_samples_per_position in validation set
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(X)
    df['position'] = y
    df['original_index'] = df.index
    
    # Group by position (you might want to bin continuous positions)
    position_bins = pd.cut(y, bins=min(8, len(np.unique(y))), labels=False)
    df['position_bin'] = position_bins
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Split for each position bin separately
    for bin_id in df['position_bin'].unique():
        bin_data = df[df['position_bin'] == bin_id]
        bin_indices = bin_data['original_index'].values
        
        if len(bin_indices) >= (min_samples_per_position * 3):  # Enough for all splits
            bin_train, bin_temp = train_test_split(
                bin_indices, test_size=(val_size + test_size), 
                random_state=42
            )
            bin_val, bin_test = train_test_split(
                bin_temp, test_size=test_size/(val_size + test_size),
                random_state=42
            )
        else:
            # For small bins, prioritize validation set
            bin_train, bin_val_test = train_test_split(
                bin_indices, test_size=min(len(bin_indices)-1, min_samples_per_position*2), 
                random_state=42
            )
            bin_val, bin_test = train_test_split(
                bin_val_test, test_size=0.5, random_state=42
            )
        
        train_indices.extend(bin_train)
        val_indices.extend(bin_val)
        test_indices.extend(bin_test)
    
    # Convert back to arrays
    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Use the manual stratification
X_train, X_val, X_test, y_train, y_val, y_test = stratified_split_positions(
    X_scaled, y_scaled, val_size=0.2, test_size=0.2, min_samples_per_position=2
)

print(f"\nFinal split:")
print(f"Training: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples") 
print(f"Test: {X_test.shape[0]} samples")

# Verify minimum samples in validation set
val_positions_original = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
min_val_samples = pd.Series(val_positions_original).value_counts().min()
print(f"Minimum samples per position in validation: {min_val_samples}")


X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val) 
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train)
y_val_tensor = torch.FloatTensor(y_val)

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
def train_model_with_validation(model, X_train, y_train, X_val, y_val, epochs=200):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs.squeeze(), y_train)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses

# Train the model
print("Starting training...")
train_losses, val_losses = train_model_with_validation(
    model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs=200
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
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training Loss')

# Predictions vs True
plt.subplot(1, 2, 2)
plt.scatter(test_true, test_predictions, alpha=0.6)
plt.plot([test_true.min(), test_true.max()], [test_true.min(), test_true.max()], 'r--', lw=2)
plt.xlabel('True Position')
plt.ylabel('Predicted Position')
plt.title(f'Predictions vs True (R² = {r2:.3f})')


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
