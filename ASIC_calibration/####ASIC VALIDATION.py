####ASIC VALIDATION

import pandas as pd

def process_validated_positions_file(filename):
    # Read the CSV, handling the comment lines and finding the actual data
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find where the data starts (after the header comments)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('TStamp'):
            data_start = i
            break
    
    # Read the data part
    df = pd.read_csv(filename, skiprows=data_start)
    
    print(f"Columns found: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Group by TStamp and filter
    valid_timestamps = []
    
    for tstamp, group in df.groupby('TStamp'):
        ch_ids = group['CH_Id'].unique()
        
        # Check criteria: must have channels 16, 17, 18 AND no more than 4 total channels
        has_required_channels = all(ch in ch_ids for ch in [16, 17, 18])
        has_four_or_less_channels = len(ch_ids) <= 4
        
        if has_required_channels and has_four_or_less_channels:
            valid_timestamps.append(tstamp)
    
    # Filter DataFrame to only include valid timestamps
    filtered_df = df[df['TStamp'].isin(valid_timestamps)].copy()
    
    # Replace timestamps with sequential index
    unique_timestamps = filtered_df['TStamp'].unique()
    timestamp_to_index = {ts: idx for idx, ts in enumerate(sorted(unique_timestamps), 1)}
    filtered_df['Index'] = filtered_df['TStamp'].map(timestamp_to_index)
    
    # Select and reorder columns
    result_df = filtered_df[['Index', 'CH_Id', 'PHA_LG']].sort_values(['Index', 'CH_Id'])
    
    return result_df

# Process the file
try:
    result = process_validated_positions_file('Validated_Positions.csv')
    
    # Display results
    print("\n" + "=" * 50)
    print("PROCESSED DATA SUMMARY")
    print("=" * 50)
    print(f"Total valid events: {result['Index'].nunique()}")
    print(f"Total rows: {len(result)}")
    print(f"Channels per event: {result.groupby('Index')['CH_Id'].count().unique()}")
    
    print("\nFirst 30 rows:")
    print(result.head(30))
    
    # Save to new CSV
    result.to_csv('processed_validated_positions.csv', index=False)
    print(f"\nResults saved to 'processed_validated_positions.csv'")
    
except Exception as e:
    print(f"Error: {e}")
    print("Trying alternative approach...")
    
    # Alternative approach - manual parsing
    with open('Validated_Positions.csv', 'r') as f:
        lines = f.readlines()
    
    # Find data lines (skip comments and header)
    data_lines = []
    for line in lines:
        if not line.startswith('//') and not line.startswith('TStamp') and line.strip():
            data_lines.append(line)
    
    # Parse manually
    data = []
    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) >= 4:  # Ensure we have enough columns
            try:
                tstamp = float(parts[0])
                ch_id = int(parts[5])  # CH_Id is 6th column (0-indexed)
                pha_lg = int(parts[7])  # PHA_LG is 8th column
                data.append({'TStamp': tstamp, 'CH_Id': ch_id, 'PHA_LG': pha_lg})
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(data)
    print(f"Manually parsed data - Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Continue with the same filtering logic...
    valid_timestamps = []    
    for tstamp, group in df.groupby('TStamp'):
        ch_ids = group['CH_Id'].unique()
        
        has_required_channels = all(ch in ch_ids for ch in [16, 17, 18])
        has_four_or_less_channels = len(ch_ids) <= 4
        
        if has_required_channels and has_four_or_less_channels:
            valid_timestamps.append(tstamp)
    
    filtered_df = df[df['TStamp'].isin(valid_timestamps)].copy()
    unique_timestamps = filtered_df['TStamp'].unique()
    timestamp_to_index = {ts: idx for idx, ts in enumerate(sorted(unique_timestamps), 1)}
    filtered_df['Index'] = filtered_df['TStamp'].map(timestamp_to_index)
    
    result_df = filtered_df[['Index', 'CH_Id', 'PHA_LG']].sort_values(['Index', 'CH_Id'])
    
    print(f"\nProcessed {len(result_df)} rows from {result_df['Index'].nunique()} valid events")
    print("First 30 rows:")
    print(result_df.head(30))
    
    result_df.to_csv('processed_validated_positions.csv', index=False)