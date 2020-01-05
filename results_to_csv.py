################################################################################
#                                                                              #
# Script to convert results to csv                                             #
#                                                                              #
# (c) Simon Wenkel, released under the Apache v2 license (see license file)    #
#                                                                              #
#                                                                              #
################################################################################



################################################################################
# import libraries                                                             #
#                                                                              #
import pickle
import numpy as np
import pandas as pd
import os
#                                                                              #
################################################################################



################################################################################
# functions                                                                    #
#                                                                              #
def main():
    files = np.sort(os.listdir("./results"))
    results = {}
    results["NumPy"] = pickle.load(open("./results/"+files[0], "rb"))
    results["CPython"] = pickle.load(open("./results/"+files[1], "rb"))
    results["PyTorchCPU"] = pickle.load(open("./results/"+files[2], "rb"))
    results["PyTorchGPU"] = pickle.load(open("./results/"+files[3], "rb"))
    results["TensorFlowCPU"] = pickle.load(open("./results/"+files[4], "rb"))

    dfList = []
    for lib in results:
        libList = [lib for j in range(200)]
        for dataset in results[lib]:
            dsList = [dataset for j in range(200)]
            dfTMP = pd.DataFrame.from_dict(results[lib][dataset])
            dfTMP.insert(loc=0, column="Library",value=libList)
            dfTMP.insert(loc=1, column="Dataset",value=dsList)
            dfList.append(dfTMP)

    resultsDF = pd.concat(dfList, sort=False)
    resultsDF = resultsDF.reset_index()
    resultsDF.drop(["index"], axis=1, inplace=True)
    resultsDF.to_csv("./results/results.csv", index=False)
#                                                                              #
################################################################################



if __name__ == "__main__":
    main()
