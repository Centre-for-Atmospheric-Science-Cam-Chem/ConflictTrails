#!/usr/bin/env python3

from process_month_emissions import process_month_emissions
import pandas as pd
import time

def test_single_month():
    """Test processing a single month to verify setup"""
    
    # Test with just January 2023
    month_start = '2023-01-01T00:00:00Z'
    output_dir = "/scratch/omg28/Data/"
    
    # Load performance model
    performance_model = pd.read_pickle('performance_and_emissions_model.pkl')
    
    print("Testing single month processing...")
    print(f"Month: {month_start}")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        result_file = process_month_emissions(
            month_start,
            output_dir=output_dir,
            performance_and_emissions_model=performance_model
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✓ Completed successfully in {processing_time:.1f} seconds")
        print(f"✓ Output file: {result_file}")
        print(f"✓ Processing speed: {1000 / processing_time:.1f} flights/second")
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"✗ Failed after {processing_time:.1f} seconds")
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_month()
