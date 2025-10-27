####ASIC VALIDATION

import pandas as pd

# File name variable
FILENAME = 'Run_Timing.csv'

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
    
    for tstamp, group in df.groupby('TStamp_us'):
        ch_ids = group['CH_Id'].unique()
        
        # Check criteria: must have channels 16, 17, 18 AND no more than 4 total channels
        has_required_channels = all(ch in ch_ids for ch in [16, 17, 18])
        has_four_or_less_channels = len(ch_ids) <= 4
        
        if has_required_channels and has_four_or_less_channels:
            valid_timestamps.append(tstamp)
    
    # Filter DataFrame to only include valid timestamps
    filtered_df = df[df['TStamp_us'].isin(valid_timestamps)].copy()
    
    # Replace timestamps with sequential index
    unique_timestamps = filtered_df['TStamp_us'].unique()
    timestamp_to_index = {ts: idx for idx, ts in enumerate(sorted(unique_timestamps), 1)}
    filtered_df['Index'] = filtered_df['TStamp_us'].map(timestamp_to_index)
    
    # Select and reorder columns - focusing on timing data (ToA_ns and ToT_ns)
    result_df = filtered_df[['Index', 'CH_Id', 'ToA_ns', 'ToT_ns']].sort_values(['Index', 'CH_Id'])
    
    return result_df

# Process the file
try:
    result = process_validated_positions_file(FILENAME)
    
    # Display results
    print("\n" + "=" * 50)
    print("PROCESSED TIMING DATA SUMMARY")
    print("=" * 50)
    print(f"File processed: {FILENAME}")
    print(f"Total valid events: {result['Index'].nunique()}")
    print(f"Total rows: {len(result)}")
    print(f"Channels per event: {result.groupby('Index')['CH_Id'].count().unique()}")
    
    print("\nFirst 30 rows:")
    print(result.head(30))
    
    # Save to new CSV
    output_filename = 'processed_timing_data.csv'
    result.to_csv(output_filename, index=False)
    print(f"\nResults saved to '{output_filename}'")
    
except Exception as e:
    print(f"Error: {e}")
    print("Trying alternative approach...")
    
    # Alternative approach - manual parsing
    with open(FILENAME, 'r') as f:
        lines = f.readlines()
    
    # Find data lines (skip comments and header)
    data_lines = []
    for line in lines:
        if not line.startswith('//') and not line.startswith('TStamp') and line.strip():
            data_lines.append(line)
    
    # Parse manually - focusing on timing data
    data = []
    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) >= 7:  # Ensure we have enough columns for timing data
            try:
                tstamp = float(parts[0])  # TStamp_us
                ch_id = int(parts[3])     # CH_Id is 4th column (0-indexed)
                toa_ns = float(parts[5])  # ToA_ns is 6th column
                tot_ns = float(parts[6])  # ToT_ns is 7th column
                data.append({'TStamp_us': tstamp, 'CH_Id': ch_id, 'ToA_ns': toa_ns, 'ToT_ns': tot_ns})
            except (ValueError, IndexError) as parse_error:
                print(f"Parse error on line: {line.strip()} - {parse_error}")
                continue
    
    df = pd.DataFrame(data)
    print(f"Manually parsed data - Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Continue with the same filtering logic...
    valid_timestamps = []    
    for tstamp, group in df.groupby('TStamp_us'):
        ch_ids = group['CH_Id'].unique()
        
        has_required_channels = all(ch in ch_ids for ch in [16, 17, 18])
        has_four_or_less_channels = len(ch_ids) <= 4
        
        if has_required_channels and has_four_or_less_channels:
            valid_timestamps.append(tstamp)
    
    filtered_df = df[df['TStamp_us'].isin(valid_timestamps)].copy()
    unique_timestamps = filtered_df['TStamp_us'].unique()
    timestamp_to_index = {ts: idx for idx, ts in enumerate(sorted(unique_timestamps), 1)}
    filtered_df['Index'] = filtered_df['TStamp_us'].map(timestamp_to_index)
    
    # Include only timing data in the result
    result_df = filtered_df[['Index', 'CH_Id', 'ToA_ns', 'ToT_ns']].sort_values(['Index', 'CH_Id'])
    
    print(f"\nProcessed {len(result_df)} rows from {result_df['Index'].nunique()} valid events")
    print("First 30 rows:")
    print(result_df.head(30))
    
    output_filename = 'processed_timing_data.csv'
    result_df.to_csv(output_filename, index=False)
    print(f"Results saved to '{output_filename}'")