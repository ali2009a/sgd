
import numpy as np
import scipy.sparse
import pandas as pd
import statsmodels.api as sm
import statsmodels
from tqdm import tqdm
import scipy.stats as stats
from numpy import genfromtxt
import subprocess
import time
def computeFrequency(data, target):

    freqs=[]
    for i in range(len(data)):
        row=data[i,:]

        counts=np.unique(row, return_counts=True)
        dic = {}
        for index, num in enumerate(counts[0]):
            dic[num] = counts[1][index]

        if 0 in dic:
            zero_freq= dic[0]
        else:
            zero_freq=0

        if 1 in dic:
            one_freq= dic[1]
        else:
            one_freq=0

        if target==0:
            frequency = zero_freq/(zero_freq+one_freq)
        elif target==1:
            frequency = one_freq/(zero_freq+one_freq)
        else:
            print("Invalid number")
        freqs.append(frequency)
    return freqs

import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt 
  
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 


def linearRegr(x,y):
    # estimating coefficients 
    b = estimate_coef(x, y) 
    print("Estimated coefficients:\nb_0 = {}  nb_1 = {}".format(b[0], b[1]))
    # plotting regression line 
    plot_regression_line(x, y, b)     


def drawHist(x):
    plt.hist(x, density=True,bins=100)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.show()


def scatterPlot(x,y):
    plt.scatter(x, y)
    plt.show()


def readData():
    dom = scipy.sparse.load_npz('realData/snpsback_gt_dominant.npz')
    dom = np.array(dom.todense()).T
    rec = scipy.sparse.load_npz('realData/snpsback_gt_recessive.npz')
    rec = np.array(rec.todense()).T
    clinicalData = pd.read_excel("realData/ClinicalData.xlsx")
    y = clinicalData["CIO_Grade"]
    
    indices = []
    for index,i in enumerate(y):
        if isinstance(i,int) and  i!=2:
            indices.append(index)


    dom = dom[indices]
    rec=  rec[indices]
    y=y[indices]
    clinicalData = clinicalData.iloc[indices]

    binarized_y=[]
    for i in y:
        if i==0 or i==1:
            binarized_y.append(1)
        elif i==3 or i==4:
            binarized_y.append(0)
    binarized_y = np.array(binarized_y)

    clinicalData = binarize_clinicalData(clinicalData)

    # clinicalData = clinicalData.drop(['index'], axis=1)
    return [dom, rec, binarized_y, clinicalData]

def binarize_clinicalData(clinicalData):
    clinicalData= clinicalData.reset_index()
    cols=clinicalData.columns[ list(range(0,6)) + list(range(20,22)) +[ 36]]
    clinicalData=clinicalData.loc[:,cols]
    clinicalData["AgeTreatmentInitiation (years)"] = clinicalData["AgeTreatmentInitiation (years)"]>=8
    clinicalData["CisplatinDose (mg/m2)"] = clinicalData["CisplatinDose (mg/m2)"]>380
    clinicalData["CisplatinDuration (days)"] = clinicalData["CisplatinDuration (days)"] >120
    return clinicalData



def computeCorreltaionFeatures(features, y):
    pvals=[]
    for col_index in tqdm(range(features.shape[1])):
        col = features[:,col_index]
        pval = computeCorrelationFeature(col,y)
        pvals.append(pval)
    return pvals

def computeCorrelationFeature(x,y):
    tab = pd.crosstab(x,y)
    # print (tab)
    if not 1 in tab.index:       
        tab.loc[1]=0

    if not 0 in tab.index:       
        tab.loc[0]=0

    # print (tab)    
    oddsratio, pvalue = stats.fisher_exact(tab)
    return pvalue



def mkdf():
    [dom, rec, y, clinicalData ] = readData()
    features =  np.load("realData/snpsback_variants.npy")

    pvals_original_dom  = genfromtxt('pvals_dom.txt', delimiter=',')
    pvals_original_rec  = genfromtxt('pvals_rec.txt', delimiter=',')
    min_pval_indices_dom= sorted(range(len(pvals_original_dom)), key=lambda i: pvals_original_dom[i])[:40]
    min_pval_indices_rec= sorted(range(len(pvals_original_rec)), key=lambda i: pvals_original_rec[i])[:40]

    min_pval_features_dom = [ features[x] for x in min_pval_indices_dom]
    min_pval_features_rec = [ features[x] for x in min_pval_indices_rec]

    df_min_40_dom = pd.DataFrame(dom[:,min_pval_indices_dom], columns= ["min_dom_"+x for x in min_pval_features_dom])
    df_min_40_rec = pd.DataFrame(rec[:,min_pval_indices_rec], columns= ["min_rec_"+x for x in min_pval_features_rec])

    # clinicalData = pd.read_excel("realData/ClinicalData.xlsx")

    df_las_61_dom = pd.DataFrame(dom[:,-61:], columns= ["dom_"+x for x in features[-61:]])
    df_las_61_rec = pd.DataFrame(rec[:,-61:], columns= ["rec_"+x for x in features[-61:]])

    binarized_y = pd.DataFrame(y, columns=["binarized_y"]) 

    result = pd.concat([ clinicalData, df_las_61_rec, df_las_61_dom, df_min_40_rec, df_min_40_dom, binarized_y], axis=1, sort=False)

    print (clinicalData.shape)
    print (df_las_61_rec.shape)
    print (df_las_61_dom.shape)
    print (df_min_40_rec.shape)
    print (df_min_40_dom.shape)
    print (binarized_y.shape)

    result = result.drop(['index'], axis=1)

    return result


#for QED
def mkdf2():
    [dom, rec, y, clinicalData ] = readData()
    features =  np.load("realData/snpsback_variants.npy")


    # clinicalData = pd.read_excel("realData/ClinicalData.xlsx")

    dom = pd.DataFrame(dom, columns=["dom_"+x for x in features])
    rec = pd.DataFrame(rec, columns=["rec_"+x for x in features])

    binarized_y = pd.DataFrame(y, columns=["binarized_y"]) 

    result = pd.concat([ clinicalData, rec, dom, binarized_y], axis=1, sort=False)
    result = result.drop(['index'], axis=1)
    return result


def getTreatmentGroups(df, indVariable):
    vals  = np.array(df[indVariable])

    controlIndexes = np.where(vals==0)[0]
    treatmentIndexes = np.where(vals==1)[0]

    return [controlIndexes, treatmentIndexes]

def ComputeCostMatrix(df, treatmentGroups, indVariable):
    controlIndexes = treatmentGroups[0]
    treatmentIndexes = treatmentGroups[1]
    weights_local = []

    confounders = df.columns[1:7]
    weights_local = np.ones((len(confounders)))
    confDF = df[confounders]
    numTreat = len(treatmentIndexes)
    numControl = len(controlIndexes)
    winLen = len(confounders)
    C = np.zeros(shape = (numTreat, numControl))


    T = np.zeros(shape = (numTreat, winLen))
    C = np.zeros(shape = (numControl, winLen))

    for i in range(numTreat):
        T[i,:] = confDF.loc[treatmentIndexes[i]].values
    
    for j in range(numControl):
        C[j,:] = confDF.loc[controlIndexes[j]].values

    T = T.reshape(T.shape[0], 1, T.shape[1])
    # TL = TL.reshape(TL.shape[0], 1, TL.shape[1])
    diff = np.abs(T-C)  
    aggregatedCost =np.sum(diff,axis=2)
    varCost = aggregatedCost / np.sum(weights_local)
    return varCost

    # for i in (range(numTreat)):
    #     for j in range(numControl):
    #         C[i,j] = computeDistance(confDF.loc[treatmentIndexes[i]].values, confDF.loc[controlIndexes[j]].values,weights_local)

    # return C


def computeDistance(row1,row2, weights_local):
    row1 = row1.astype(int)
    row2 = row2.astype(int)    
    diff  = row1 - row2
    diff = diff*weights_local
    diff = diff[~np.isnan(diff)]
    return np.linalg.norm(diff)/len(diff)


def performMatching(C):

    r,c = C.shape
    with open('matrix.txt', 'w') as f:
        f.write("{} {}\n".format(r,c))
        for i in range(0,r):
            for j in range(0,c):
                f.write( "{} ".format(C[i][j]))
            f.write("\n")

    command = "hungarian/test"
    run_cmd(command)
    
    costs = []
    with open('matching.txt', 'r') as f:
        indexes = []
        for line in f:
            words = line.rstrip('\n').split(',')
            L = int(words[0])
            R = int(words[1])
            if R!= -1:
                pair = (L,R)
                indexes.append(pair)
                costs.append(C[L,R])

    costs = np.array(costs)
    passedPairs = [pair for idx, pair in enumerate(indexes) if costs[idx]< 0.3 ]            
    # m = Munkres()
    # indexes = m.compute(C)
    return passedPairs


def run_cmd(cmd, working_directory=None):
    if working_directory!= None:
        try:
            output = subprocess.check_output(cmd,shell=True,cwd=working_directory)
            print ("output:"+output)
        except:
            print ("failed:"+cmd)
            # pass
    else:
        try:
            output = subprocess.check_output(cmd,shell=True)
            print ((output))
        except:
            print ("failed:"+cmd)
            # pass


def computePValue_MCNmar(X,Y):
    # X = np.array(X).astype(str)
    # Y = 
    X = pd.Categorical(X, categories=[0,1])
    Y = pd.Categorical(Y, categories=[0,1])
    tab = pd.crosstab(X,Y, dropna=False)
    res= statsmodels.stats.contingency_tables.mcnemar(tab)
    pVal = res.pvalue
    return pVal


def getTargetValues(df, treatmentGroups, indexes):
    controlIndexes = treatmentGroups[0]
    treatmentIndexes = treatmentGroups[1]
    T_indices = [  treatmentIndexes[i[0]]  for i in indexes]
    memtotT = df.loc[T_indices]["binarized_y"]
    C_indices = [  controlIndexes[i[1]]  for i in indexes]
    memtotC = df.loc[C_indices]["binarized_y"]
    return [memtotC, memtotT]


def QED():    
    df = mkdf2()

    features =  np.load("realData/snpsback_variants.npy")
    dom_features = ["dom_"+x for x in features]
    rec_features = ["rec_"+x for x in features]
    indVariables = dom_features + rec_features

    pVals = []
    for  index, indVariable in enumerate(tqdm(indVariables)):
    # for index, indVariable in enumerate(["dom_1_rs35699260"]):
            treatmentGroups = getTreatmentGroups(df,indVariable) #alternative
            if len(treatmentGroups[0])==0 or  len(treatmentGroups[1])==0:
                pVals.append("NA:NET")
                continue 
            C= ComputeCostMatrix(df, treatmentGroups, indVariable)
            matchedPairs = performMatching(C)
            if len(matchedPairs)==0:
                pVals.append("NA:NEP")
                continue             
            targetValues = getTargetValues(df,treatmentGroups, matchedPairs)
            pval = computePValue_MCNmar(targetValues[0], targetValues[1])            
            pVals.append(pval)
    return (pVals)


