#!/usr/bin/env python3

import pandas as pd
import time

def test_data_loading():
    """Test loading the data files to see if that's the bottleneck"""
    
    output_dir = "/scratch/omg28/Data/"
    start_time_simple_loop = "2023-01-01"
    stop_time_simple_loop = "2023-01-31"
    
    print("Testing data loading...")
    
    file_path = f'{output_dir}/{start_time_simple_loop}_to_{stop_time_simple_loop}_filtered.pkl'
    print(f"Loading: {file_path}")
    
    start_time = time.time()
    
    try:
        # Load just first 1000 flights like in the original code
        monthly_flights = pd.read_pickle(file_path)[0:1000]
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        print(f"✓ Data loaded successfully in {loading_time:.1f} seconds")
        print(f"✓ Number of flights: {len(monthly_flights)}")
        print(f"✓ Columns: {list(monthly_flights.columns)}")
        print(f"✓ Unique typecodes: {len(monthly_flights['typecode'].unique())}")
        
        # Show a sample row
        print("\nSample flight data:")
        print(monthly_flights.head(1).to_string())
        
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()
