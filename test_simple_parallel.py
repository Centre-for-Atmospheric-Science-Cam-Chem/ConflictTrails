#!/usr/bin/env python3

from process_month_emissions import process_month_emissions
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
import os

def process_single_month(args):
    """Wrapper function for processing a single month"""
    start_time_str_loop, output_dir, performance_and_emissions_model = args
    try:
        print(f"Starting month: {start_time_str_loop.strftime('%Y-%m')}")
        start_time = time.time()
        
        result = process_month_emissions(
            start_time_str_loop,
            output_dir=output_dir,
            performance_and_emissions_model=performance_and_emissions_model
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Completed month {start_time_str_loop.strftime('%Y-%m')} in {processing_time:.1f} seconds")
        return start_time_str_loop.strftime('%Y-%m'), result, processing_time
    except Exception as e:
        print(f"Error processing month {start_time_str_loop.strftime('%Y-%m')}: {e}")
        import traceback
        traceback.print_exc()
        return start_time_str_loop.strftime('%Y-%m'), None, None

if __name__ == "__main__":
    # Test with just 3 months
    start_time_str = '2023-01-01T00:00:00Z'
    stop_time_str = '2023-03-31T23:59:59Z'  # Just Jan, Feb, Mar
    output_dir = "/scratch/omg28/Data/"

    # Load performance model once
    performance_and_emissions_model = pd.read_pickle('performance_and_emissions_model.pkl')
    
    # Generate month ranges
    month_ranges = list(pd.date_range(start=pd.to_datetime(start_time_str), 
                                     end=pd.to_datetime(stop_time_str), 
                                     freq='MS', tz='UTC'))
    
    # Prepare arguments for parallel processing
    month_args = [
        (start_time_str_loop, output_dir, performance_and_emissions_model)
        for start_time_str_loop in month_ranges
    ]
    
    # Use 3 processes for 3 months
    total_cpus = cpu_count()
    max_month_processes = min(len(month_ranges), total_cpus)
    
    print(f"Testing with {len(month_ranges)} months using {max_month_processes} parallel processes")
    print(f"Total CPU cores: {total_cpus}")
    print("=" * 60)
    
    overall_start_time = time.time()
    
    # Process months in parallel
    with Pool(processes=max_month_processes) as pool:
        results = pool.map(process_single_month, month_args)
    
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    # Print summary
    print("=" * 60)
    print(f"TEST COMPLETE")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    successful_months = []
    failed_months = []
    
    for month, result, proc_time in results:
        if result is not None:
            successful_months.append((month, proc_time))
            print(f"✓ {month}: {result}")
        else:
            failed_months.append(month)
            print(f"✗ {month}: FAILED")
    
    print(f"\nSuccessful: {len(successful_months)}/{len(month_ranges)} months")
    if failed_months:
        print(f"Failed months: {', '.join(failed_months)}")
    
    if successful_months:
        avg_time = sum(proc_time for _, proc_time in successful_months) / len(successful_months)
        print(f"Average processing time per month: {avg_time:.1f} seconds")
        
        # Calculate flights per second per month
        avg_flights_per_second_per_month = 1000 / avg_time
        print(f"Average flights per second per month: {avg_flights_per_second_per_month:.1f}")
        
        # Estimate performance for full year
        estimated_total_time = avg_time * 12 / max_month_processes
        print(f"Estimated time for full year: {estimated_total_time:.1f} seconds ({estimated_total_time/60:.1f} minutes)")
