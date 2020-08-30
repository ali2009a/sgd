
import numpy as np
import scipy.sparse
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
import scipy.stats as stats
from numpy import genfromtxt


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
    

    binarized_y=[]
    for i in y:
        if i==0 or i==1:
            binarized_y.append(1)
        elif i==3 or i==4:
            binarized_y.append(0)
    binarized_y = np.array(binarized_y)

    clinicalData = clinicalData.iloc[indices]
    clinicalData= clinicalData.reset_index()
    cols=clinicalData.columns[ list(range(0,25))+[35]]
    clinicalData=clinicalData.loc[:,cols]
    return [dom, rec, binarized_y, clinicalData]


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



