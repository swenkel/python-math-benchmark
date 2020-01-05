################################################################################
#                                                                              #
# Script to run PyTorch GPU benchmarks                                         #
#                                                                              #
# (c) Simon Wenkel, released under the Apache v2 license (see license file)    #
#                                                                              #
#                                                                              #
################################################################################



################################################################################
# import libraries                                                             #
#                                                                              #
import pickle
import time
from tqdm import tqdm
import torch
#                                                                              #
################################################################################



################################################################################
# function dict                                                                #
#                                                                              #
functions = {}
functions["sin"] = torch.sin
functions["cos"] = torch.cos
functions["tan"] = torch.tan
functions["asin"] = torch.asin
functions["acos"] = torch.acos
functions["atan"] = torch.atan
functions["exp"] = torch.exp
functions["sinh"] = torch.sinh
functions["cosh"] = torch.cosh
functions["tanh"] = torch.tanh
functions["abs"] = torch.abs
functions["ceil"] = torch.ceil
functions["floor"] = torch.floor
functions["sqrt"] = torch.sqrt
#                                                                              #
################################################################################



################################################################################
# functions                                                                    #
#                                                                              #
def list_of_items(listSize:int,
                  functions:dict,
                  device:torch.device):
    itemList = torch.rand(listSize, device=device)
    results = {}
    for function in functions:
        results[function]  = {}
        counter = 0
        for i in range(200):
            startTime = time.time()
            functions[function](itemList)
            results[function][counter] = time.time()-startTime
            counter += 1
    return results



def array_of_items(arraySize:int,
                   functions:dict,
                   device:torch.device):
    matrix = torch.rand((arraySize,arraySize), device=device)
    results = {}
    for function in functions:
        results[function]  = {}
        counter = 0
        for iteration in range(200):
            startTime = time.time()
            functions[function](matrix)
            results[function][counter] = time.time()-startTime
            counter += 1
    return results



def main():
    start = time.time()
    results = {}
    device = torch.device('cuda')
    for i in tqdm([1,10,100,1000,10000,100000,1000000]):
        results["List_"+str(i)] = list_of_items(i, functions, device)
    for i in tqdm([1,10,100,1000,10000]):
        results["Matrix_"+str(i)] = array_of_items(i, functions, device)
    pickle.dump(results,open("./results/PyTorch_gpu.pkl", "wb"))
#                                                                              #
################################################################################



if __name__ == "__main__":
    main()
