################################################################################
#                                                                              #
# Script to run TensorFlow CPU benchmarks                                      #
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
import tensorflow as tf
#                                                                              #
################################################################################



################################################################################
# function dict                                                                #
#                                                                              #
functions = {}
functions["sin"] = tf.sin
functions["cos"] = tf.cos
functions["tan"] = tf.tan
functions["asin"] = tf.asin
functions["acos"] = tf.acos
functions["atan"] = tf.atan
functions["exp"] = tf.exp
functions["sinh"] = tf.sinh
functions["cosh"] = tf.cosh
functions["tanh"] = tf.tanh
functions["abs"] = tf.abs
functions["ceil"] = tf.math.ceil
functions["floor"] = tf.math.floor
functions["sqrt"] = tf.sqrt
#                                                                              #
################################################################################



################################################################################
# functions                                                                    #
#                                                                              #
def list_of_items(listSize:int,
                  functions:dict):
    itemList = tf.random.uniform([listSize])
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
    matrix = tf.random.uniform([arraySize,arraySize])
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
    device = tf.device('cpu')
    for i in tqdm([1,10,100,1000,10000,100000,1000000]):
        results["List_"+str(i)] = list_of_items(i, functions)
    for i in tqdm([1,10,100,1000,10000]):
        results["Matrix_"+str(i)] = array_of_items(i, functions)
    pickle.dump(results,open("./results/TensorFlow_cpu.pkl", "wb"))
#                                                                              #
################################################################################



if __name__ == "__main__":
    main()
