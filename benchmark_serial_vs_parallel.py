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
        start_time = time.time()
        
        result = process_month_emissions(
            start_time_str_loop,
            output_dir=output_dir,
            performance_and_emissions_model=performance_and_emissions_model
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        return start_time_str_loop.strftime('%Y-%m'), result, processing_time
    except Exception as e:
        print(f"Error processing month {start_time_str_loop.strftime('%Y-%m')}: {e}")
        return start_time_str_loop.strftime('%Y-%m'), None, None

def run_serial_test(month_ranges, output_dir, performance_and_emissions_model):
    """Run months serially for comparison"""
    print("Running SERIAL test...")
    start_time = time.time()
    
    results = []
    for month in month_ranges:
        args = (month, output_dir, performance_and_emissions_model)
        result = process_single_month(args)
        results.append(result)
        print(f"  Completed {result[0]} in {result[2]:.1f}s")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return results, total_time

def run_parallel_test(month_ranges, output_dir, performance_and_emissions_model):
    """Run months in parallel"""
    print("Running PARALLEL test...")
    
    month_args = [
        (start_time_str_loop, output_dir, performance_and_emissions_model)
        for start_time_str_loop in month_ranges
    ]
    
    max_month_processes = min(len(month_ranges), cpu_count())
    
    start_time = time.time()
    
    with Pool(processes=max_month_processes) as pool:
        results = pool.map(process_single_month, month_args)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return results, total_time

if __name__ == "__main__":
    # Test with just 2 months to keep it quick
    start_time_str = '2023-01-01T00:00:00Z'
    stop_time_str = '2023-02-28T23:59:59Z'  # Just Jan, Feb
    output_dir = "/scratch/omg28/Data/"

    # Load performance model once
    performance_and_emissions_model = pd.read_pickle('performance_and_emissions_model.pkl')
    
    # Generate month ranges
    month_ranges = list(pd.date_range(start=pd.to_datetime(start_time_str), 
                                     end=pd.to_datetime(stop_time_str), 
                                     freq='MS', tz='UTC'))
    
    print(f"Benchmark: Serial vs Parallel processing")
    print(f"Testing with {len(month_ranges)} months")
    print(f"Available CPUs: {cpu_count()}")
    print("=" * 60)
    
    # Run serial test
    serial_results, serial_time = run_serial_test(month_ranges, output_dir, performance_and_emissions_model)
    
    print("=" * 60)
    
    # Run parallel test  
    parallel_results, parallel_time = run_parallel_test(month_ranges, output_dir, performance_and_emissions_model)
    
    # Compare results
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"Serial processing:   {serial_time:.1f} seconds")
    print(f"Parallel processing: {parallel_time:.1f} seconds")
    
    if parallel_time > 0:
        speedup = serial_time / parallel_time
        print(f"Speedup: {speedup:.2f}x")
        
        # Check if we got the expected speedup
        expected_speedup = min(len(month_ranges), cpu_count())
        efficiency = speedup / expected_speedup * 100
        print(f"Parallel efficiency: {efficiency:.1f}% (expected ~{expected_speedup}x speedup)")
    
    # Verify both produced the same number of successful results
    serial_successful = sum(1 for _, result, _ in serial_results if result is not None)
    parallel_successful = sum(1 for _, result, _ in parallel_results if result is not None)
    
    print(f"Serial successful months: {serial_successful}")
    print(f"Parallel successful months: {parallel_successful}")
    
    if serial_successful == parallel_successful:
        print("✓ Both methods processed the same number of months successfully")
    else:
        print("⚠ Different success rates between serial and parallel")
