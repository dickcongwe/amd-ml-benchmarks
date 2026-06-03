#!/usr/bin/env python3
"""
AMD ML Benchmark Suite
Comprehensive GPU benchmarks for ROCm
"""

import torch
import time
import argparse
import json


class AMDBenchmark:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        if self.device.type == 'cuda':
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        else:
            self.gpu_name = 'CPU'
            self.gpu_mem = 0
    
    def bench_matmul(self, size=4096, iterations=100):
        print(f"\nMatMul {size}x{size}:")
        a = torch.randn(size, size, device=self.device)
        b = torch.randn(size, size, device=self.device)
        
        # Warmup
        for _ in range(10):
            torch.mm(a, b)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        tflops = 2 * size**3 * iterations / elapsed / 1e12
        print(f"  Time: {elapsed/iterations*1000:.2f}ms | {tflops:.1f} TFLOPS")
        return tflops
    
    def bench_bandwidth(self, size=100_000_000, iterations=100):
        print(f"\nMemory Bandwidth:")
        a = torch.randn(size, device=self.device)
        b = torch.empty_like(a)
        
        for _ in range(10):
            b.copy_(a)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            b.copy_(a)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        bw = 2 * size * 4 * iterations / elapsed / 1e9
        print(f"  Bandwidth: {bw:.1f} GB/s")
        return bw
    
    def run_all(self, iterations=100):
        print("=" * 50)
        print(f"AMD ML Benchmark Suite")
        print(f"GPU: {self.gpu_name} ({self.gpu_mem:.1f} GB)")
        print("=" * 50)
        
        self.results['gpu'] = self.gpu_name
        self.results['matmul_fp32'] = self.bench_matmul(iterations=iterations)
        self.results['bandwidth'] = self.bench_bandwidth(iterations=iterations)
        
        print("\n" + "=" * 50)
        print("Benchmark Complete!")
        
        with open('benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("Results saved to benchmark_results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    
    bench = AMDBenchmark()
    bench.run_all(iterations=args.iterations)
