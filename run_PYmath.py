################################################################################
#                                                                              #
# Script to run CPython math benchmarks                                        #
#                                                                              #
# (c) Simon Wenkel, released under the Apache v2 license (see license file)    #
#                                                                              #
#                                                                              #
################################################################################



################################################################################
# import libraries                                                             #
#                                                                              #
import pickle
import math
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
functions["sin"] = math.sin
functions["cos"] = math.cos
functions["tan"] = math.tan
functions["asin"] = math.asin
functions["acos"] = math.acos
functions["atan"] = math.atan
functions["exp"] = math.exp
functions["sinh"] = math.sinh
functions["cosh"] = math.cosh
functions["tanh"] = math.tanh
functions["abs"] = abs
functions["ceil"] = math.ceil
functions["floor"] = math.floor
functions["sqrt"] = math.sqrt
#                                                                              #
################################################################################



################################################################################
# functions                                                                    #
#                                                                              #
def list_of_items(listSize:int,
                  functions:dict):
    itemList = [random.random() for i in range(listSize)]
    results = {}
    for function in functions:
        results[function]  = {}
        counter = 0
        for i in range(200):
            startTime = time.time()
            for item in itemList:
                functions[function](item)
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
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    functions[function](matrix[i,j])
            results[function][counter] = time.time()-startTime
            counter += 1
    return results



def main():
    start = time.time()
    results = {}
    for i in tqdm([1,10,100,1000,10000,100000]):
        results["List_"+str(i)] = list_of_items(i, functions)
    for i in tqdm([1,10,100,1000]):
        results["Matrix_"+str(i)] = array_of_items(i, functions)
    pickle.dump(results,open("./results/PyMath.pkl", "wb"))
#                                                                              #
################################################################################



if __name__ == "__main__":
    main()
