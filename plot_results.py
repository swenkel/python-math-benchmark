################################################################################
#                                                                              #
# Script to plot results                                                       #
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
import matplotlib.pyplot as plt
import seaborn as sns
import os
#                                                                              #
################################################################################



################################################################################
# functions                                                                    #
#                                                                              #
def individualBoxPlots(df:pd.DataFrame,
                       library:str,
                       dataset:str):
    plt.figure(figsize=(10,8))
    df.boxplot(notch=True)
    plt.xlabel("Function")
    plt.ylabel("Runtime [s]")
    plt.title(library+" on "+dataset)
    plt.savefig("./graphics/"+library+"_"+dataset+".png", bbox_inces="tight")
    plt.savefig("./graphics/"+library+"_"+dataset+".pdf", bbox_inces="tight")
    plt.close()



def libBoxPlots(df:pd.DataFrame,
                library:str,
                setType:str,
                scale:str = "linear"):
    plt.figure(figsize=(13,12))
    sns.boxplot(x="Dataset", y="Runtime [s]", data=df, notch=True)
    plt.title(library+" on "+setType+" ("+scale+" scale)")
    plt.yscale(scale)
    plt.grid(which="both")
    plt.savefig("./graphics/"+library+"_"+setType+"_"+scale+".png",
                bbox_inces="tight")
    plt.savefig("./graphics/"+library+"_"+setType+"_"+scale+".pdf",
                bbox_inces="tight")
    plt.grid()
    plt.close()



def comparingLibraries(df:pd.DataFrame,
                       dataset:str,
                       scale:str = "linear"):
    plt.figure(figsize=(13,12))
    sns.boxplot(x="Dataset", y="Runtime [s]", hue="Library", data=df, notch=True)
    plt.yscale(scale)
    plt.grid(which="both")
    plt.savefig("./graphics/library_comparison_"+dataset+"_"+scale+".png",
                bbox_inces="tight")
    plt.savefig("./graphics/library_comparison_"+dataset+"_"+scale+".pdf",
                bbox_inces="tight")
    plt.close()



def main():
    os.makedirs("./graphics/", exist_ok=True)
    resultsDF = pd.read_csv("./results/results.csv")
    libraries = resultsDF["Library"].unique()
    listDFs = []
    MatrixDFs = []
    for lib in libraries:
        libTmpDF = resultsDF[resultsDF["Library"] == lib].copy(deep=True)
        libTmpDF.drop(["Library"], axis=1, inplace=True)
        tmpDFList = []
        tmpDFMatrix = []
        for dataset in libTmpDF["Dataset"].unique():
            individualBoxPlots(libTmpDF[libTmpDF["Dataset"] == dataset],
                               lib,
                               dataset)
            if "List" in dataset:
                tmpl = libTmpDF[libTmpDF["Dataset"] == dataset].values[:,1:].flatten()
                tmpd = [dataset for i in range(len(tmpl))]
                tmp = np.vstack((tmpd,tmpl)).T
                tmpDFList.append(pd.DataFrame(tmp, columns=["Dataset", "Runtime [s]"]))

            elif "Matrix" in dataset:
                tmpl = libTmpDF[libTmpDF["Dataset"] == dataset].values[:,1:].flatten()
                tmpd = [dataset for i in range(len(tmpl))]
                tmp = np.vstack((tmpd,tmpl)).T
                tmpDFMatrix.append(pd.DataFrame(tmp, columns=["Dataset", "Runtime [s]"]))
        ListRes = pd.concat(tmpDFList, sort=False)
        ListRes = ListRes.reset_index()
        ListRes.drop(["index"], axis=1, inplace=True)
        ListRes.insert(loc=0, column="Library",value=[lib for i in range(len(ListRes))])
        ListRes.to_csv("./results/"+lib+"_list.csv", index=False)
        listDFs.append(ListRes)
        MatrixRes = pd.concat(tmpDFMatrix, sort=False)
        MatrixRes = MatrixRes.reset_index()
        MatrixRes.drop(["index"], axis=1, inplace=True)
        MatrixRes.insert(loc=0, column="Library",value=[lib for i in range(len(MatrixRes))])
        MatrixRes.to_csv("./results/"+lib+"_matrix.csv", index=False)
        MatrixDFs.append(MatrixRes)
        libBoxPlots(ListRes,lib,"List",scale="linear")
        libBoxPlots(ListRes,lib,"List",scale="log")
        libBoxPlots(MatrixRes,lib,"Matrix", scale="linear")
        libBoxPlots(MatrixRes,lib,"Matrix", scale="log")
    lDFs = pd.concat(listDFs, sort=False)
    lDFs = lDFs.reset_index()
    lDFs.drop(["index"], axis=1, inplace=True)
    lDFs.to_csv("./results/List_comparison.csv")
    comparingLibraries(lDFs, "List", scale="linear")
    comparingLibraries(lDFs, "List",scale="log")
    mDFs = pd.concat(MatrixDFs, sort=False)
    mDFs = mDFs.reset_index()
    mDFs.drop(["index"], axis=1, inplace=True)
    mDFs.to_csv("./results/Matrix_comparison.csv")
    comparingLibraries(mDFs, "Matrix", scale="linear")
    comparingLibraries(mDFs, "Matrix",scale="log")
#                                                                              #
################################################################################



if __name__ == "__main__":
    main()
