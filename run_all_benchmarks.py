################################################################################
#                                                                              #
# Script to run all benchmarks                                                 #
#                                                                              #
# (c) Simon Wenkel, released under the Apache v2 license (see license file)    #
#                                                                              #
#                                                                              #
################################################################################



################################################################################
# import libraries                                                             #
#                                                                              #
import os
import time
globalStartTime = time.time()
from run_PYmath import main as PyMathBenchmark
from run_NumPy import main as NumPyBenchmark
from run_PyTorch_cpu import main as PyTorchCPUBechmark
from run_PyTorch_gpu import main as PyTorchGPUBechmark
from run_TensorFlow_cpu import main as TFCPUBenchmark
#                                                                              #
################################################################################



################################################################################
# functions                                                                    #
#                                                                              #

def main():
    print("=" * 80)
    print("Comparing different math libraries for Python")
    os.makedirs("./results/", exist_ok=True)
    print("*" * 40)
    print("Run native CPython benchmark")
    subStartTime = time.time()
    PyMathBenchmark()
    print("\n Finished in {:.2f} sec".format(time.time()-subStartTime))
    print("*" * 40)
    print("Run NumPy benchmark")
    subStartTime = time.time()
    NumPyBenchmark()
    print("*" * 40)
    print("\n Finished in {:.2f} sec".format(time.time()-subStartTime))
    print("Run PyTorch (CPU) benchmark")
    subStartTime = time.time()
    PyTorchCPUBechmark()
    print("\n Finished in {:.2f} sec".format(time.time()-subStartTime))
    print("*" * 40)
    print("Run PyTorch (GPU) benchmark")
    subStartTime = time.time()
    PyTorchGPUBechmark()
    print("\n Finished in {:.2f} sec".format(time.time()-subStartTime))
    print("*" * 40)
    print("Run TensorFlow (CPU) benchmark")
    subStartTime = time.time()
    TFCPUBenchmark()
    print("\n Finished in {:.2f} sec".format(time.time()-subStartTime))
    print("*" * 40)
    print("Total runtime: {:.2f} min".format((time.time()-globalStartTime)/60))
    print("=" * 80)
#                                                                              #
################################################################################



if __name__ == "__main__":
    main()
