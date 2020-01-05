################################################################################
#                                                                              #
# Script to run NumPy benchmarks                                               #
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
import random
import numpy as np
from tqdm import tqdm
#                                                                              #
################################################################################



################################################################################
# function dict                                                                #
#                                                                              #
functions = {}
functions["sin"] = np.sin
functions["cos"] = np.cos
functions["tan"] = np.tan
functions["asin"] = np.arcsin
functions["acos"] = np.arccos
functions["atan"] = np.arctan
functions["exp"] = np.exp
functions["sinh"] = np.sinh
functions["cosh"] = np.cosh
functions["tanh"] = np.tanh
functions["abs"] = np.abs
functions["ceil"] = np.ceil
functions["floor"] = np.floor
functions["sqrt"] = np.sqrt
#                                                                              #
################################################################################



################################################################################
# functions                                                                    #
#                                                                              #
def list_of_items(listSize:int,
                  functions:dict):
    itemList = np.random.random(listSize)
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
                   functions:dict):
    matrix = np.random.random((arraySize,arraySize))
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
    for i in tqdm([1,10,100,1000,10000,100000,1000000]):
        results["List_"+str(i)] = list_of_items(i, functions)
    for i in tqdm([1,10,100,1000,10000]):
        results["Matrix_"+str(i)] = array_of_items(i, functions)
    pickle.dump(results,open("./results/NumPy.pkl", "wb"))
#                                                                              #
################################################################################



if __name__ == "__main__":
    main()
