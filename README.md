# 📊 AMD ML Benchmarks

Comprehensive ML benchmark suite for AMD GPUs with ROCm support.

## Benchmarks

- **Matrix Operations** - GEMM, convolutions, attention
- **Training Throughput** - Images/sec, sequences/sec
- **Inference Latency** - Single request, batch processing
- **Memory Bandwidth** - Read/write, copy operations
- **Power Efficiency** - Performance per watt

## Usage

```bash
# Run all benchmarks
python benchmark.py --all

# Specific framework
python benchmark.py --framework pytorch
python benchmark.py --framework tensorflow
python benchmark.py --framework onnx

# Custom config
python benchmark.py --model resnet50 --batch-size 64 --iterations 1000
```

## Sample Results

```
=== AMD ROCm ML Benchmark v1.0 ===
GPU: AMD Radeon RX 7900 XTX (24GB)
ROCm: 5.7.1 | Driver: 6.1.5

PyTorch Benchmarks:
  FP32 MatMul (4096x4096): 18.2 TFLOPS
  FP16 MatMul (4096x4096): 36.4 TFLOPS
  ResNet-50 Training: 1,245 img/s
  BERT Inference: 3.2ms/seq

TensorFlow Benchmarks:
  FP32 MatMul (4096x4096): 17.8 TFLOPS
  FP16 MatMul (4096x4096): 35.6 TFLOPS
  ResNet-50 Training: 1,180 img/s

ONNX Runtime:
  ResNet-50 Inference: 1.8ms
  BERT Inference: 2.9ms/seq
```

## License

MIT
