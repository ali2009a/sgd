import pandas as pd
from itertools import combinations, chain
from math import factorial
from tqdm import tqdm
import numpy as np
from heapq import heappush, heappop


# def readData():
#     features = pd.read_csv("data/features.csv")
#     features = features.iloc[:,:-5]
#     clinicalData  =pd.read_excel("data/ClinicalData.xlsx")
#     features["outcome"] = clinicalData["Cardiotoxicity"]
#     features=features.set_index("index")
#     return features


# def readData():
#     header=  list(range(1,101))
#     header2 = [str(x) for x in header]
#     features = pd.read_csv("data/X.csv", header=None , names=header2)
#     outcome = pd.read_csv("data/Y.csv", header=None, names=["outcome"])
#     result = pd.concat([features, outcome], axis=1, sort=False)
#     # features = features.iloc[:,:-5]
#     # clinicalData  =pd.read_excel("data/ClinicalData.xlsx")
#     # features["outcome"] = clinicalData["Cardiotoxicity"]
#     # features=features.set_index("index")
#     return result

def readData():
    data_root= "data3_toy"

    # header=  list(range(0,100))
    header=  list(range(0,4))
    header2 = [str(x) for x in header]
    features = pd.read_csv("{}/X.csv".format(data_root), header=None , names=header2)
    outcome = pd.read_csv("{}/Y.csv".format(data_root), header=None, names=["outcome"])
    result = pd.concat([features, outcome], axis=1, sort=False)

    feat_score= pd.read_csv("{}/S1.csv".format(data_root), header=None)
    feat_similarity= pd.read_csv("{}/S2.csv".format(data_root), header=None)
    # feat_similarity = 1-feat_dissimilarity
    feat_similarity = np.array(feat_similarity)
    # features = features.iloc[:,:-5]
    # clinicalData  =pd.read_excel("data/ClinicalData.xlsx")
    # features["outcome"] = clinicalData["Cardiotoxicity"]
    # features=features.set_index("index")
    X=result.iloc[:,:-1]
    Y=result.iloc[:,-1:]
    return [X, Y, feat_score, feat_similarity]





class EquitySelector():

    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value

    def covers(self, data):
        column_data = data[self.attribute].to_numpy()
        if pd.isnull(self.value):
            return pd.isnull(column_data)
        return column_data == self.value

    def __repr__(self):
        query=""
        if np.isnan(self.value):
            query = self.attribute + ".isnull()"
        else:
            query = str(self.attribute) + "==" + str(self.value)
        return query    

    def __lt__(self, other):
        return repr(self) < repr(other)

# class BinaryTarget():
#     def __init__(self, attribute, value):
#         self.attribute = attribute
#         self.value = value

def createTarget(attribute, value):
    selector = EquitySelector(attribute, value)
    return selector


def createSelectors(data, ignore=[]):
    selectors = []
    original_features = []
    sg_to_index = {}
    counter=0
    for attr_name in [x for x in data if x not in ignore]:
        for val in pd.unique(data[attr_name]):
            selector = EquitySelector(attr_name, val)
            selectors.append(selector)
            original_features.append(int(attr_name))
            sg_to_index[selector] = counter
            counter=counter+1
    return [selectors, original_features, sg_to_index]    


def createSearchSpace(selectors, depth):
    def binomial(x, y):
        try:
            binom = factorial(x) // factorial(y) // factorial(x - y)
        except ValueError:
            binom = 0
        return binom
    searchSpace = chain.from_iterable(combinations(selectors, r) for r in range(1, depth + 1))
    length = sum(binomial(len(selectors), k) for k in range(1, depth + 1))
    return [searchSpace, length]


class Conjuction:
    def __init__(self, selectors):
        self.selectors = selectors

    def covers(self, data):
        # empty description ==> return a list of all '1's
        if not self.selectors:
            return np.full(len(data), True, dtype=bool)
        # non-empty description
        return np.all([sel.covers(data) for sel in self.selectors], axis=0)


    def __repr__(self, open_brackets="", closing_brackets="", and_term=" AND "):
        attrs = sorted(str(sel) for sel in self.selectors)
        return "".join((open_brackets, and_term.join(attrs), closing_brackets))

    def __lt__(self, other):
        return repr(self) < repr(other)


def add_if_required(result, sg, quality, result_set_size, check_for_duplicates=False):
    # if quality > task.min_quality:
    # if check_for_duplicates and (quality, sg) in result:
    #     return
    if len(result) < result_set_size:
        heappush(result, (quality, sg))
    elif quality > result[0][0]:
        heappop(result)
        heappush(result, (quality, sg))


def computeScore(sg_vector, outcome_vector, measure):
    n=len(sg_vector)
    sg_vector = sg_vector.astype(int)
    outcome_vector = outcome_vector.astype(int)
    tab = pd.crosstab(sg_vector,outcome_vector)
    if not 1 in tab.index:       
        tab.loc[1]=0


    TP= n11 = tab.loc[1][1]
    FP= n10 = tab.loc[1][0]
    FN= n01 = tab.loc[0][1]
    TN= n00 = tab.loc[0][0]
    N= tab.loc[0][0]+tab.loc[0][1]
    P= tab.loc[1][0]+tab.loc[1][1]
    F= tab.loc[0][0]+tab.loc[1][0]
    T= tab.loc[0][1]+tab.loc[1][1]

    e=1
    if measure=="accuracy":
        quality = (n11+n00)/n
    elif measure=="oddsRatio":
        quality = (n00*n11)/(n10*n01)
    elif measure=="colligation":
        quality= ( n11*n00 - n10*n01 )/( n11*n00 + n10*n01 + e )
    elif measure=="goodman":
        quality = 1- ((min(n11,n10)+min(n00,n01))/(min(n01,n10)))
    elif measure=="f1":
        quality = (2*n11)/(n10+n01)
    elif measure == "new":
        quality = ((TP*TN)-(FP*FN))/(np.sqrt(T*F*P*N)+e)
    return quality

def computeQuality(X, Y, measure=""):
    X = X.astype(int)
    Y = Y.astype(int)
    tab = pd.crosstab(X,Y)
    
    if not 1 in tab.index:       
        tab.loc[1]=0
    if not 0 in tab.index:       
        tab.loc[0]=0    

    tab = tab+1
    TP= n11 = tab.loc[1][1]
    FP= n10 = tab.loc[1][0]
    FN= n01 = tab.loc[0][1]
    TN= n00 = tab.loc[0][0]
    N= n0b=tab.loc[0][0]+tab.loc[0][1]
    P= n1b= tab.loc[1][0]+tab.loc[1][1]
    F= nb0= tab.loc[0][0]+tab.loc[1][0]
    T= nb1=tab.loc[0][1]+tab.loc[1][1]
    quality= ( n11*n00 - n10*n01 )/np.sqrt( n1b*n0b * nb1*nb0)
    return np.abs(quality)


def simpleSearch(target, selectors, data, measure):
    searchSpace = createSearchSpace(selectors,2)
    # print (searchSpace[1])
    # searchSpace = searchSpace[0]
    # print(type(searchSpace))
    tqdm_searchSpace = tqdm(searchSpace[0],total=searchSpace[1])
    result = []
    for i, selectors_one_point in enumerate(tqdm_searchSpace):
        sg = Conjuction(selectors_one_point)
        sg_vector = sg.covers(data)
        outcome_vector = target.covers(data)
        quality = computeScore(sg_vector, outcome_vector, measure)
        # result.append((quality,selectors_one_point))
        add_if_required(result, sg, quality, 10)
    return result


def beamSearch(target, selectors, data, measure, max_depth=2, beam_width=5, result_set_size=5):
    beam = [(0, Conjuction([]))]
    last_beam = None

    depth = 0
    while beam != last_beam and depth < max_depth:
        last_beam = beam.copy()
        print("last_beam size: {}, depth: {}".format(len(last_beam), depth))
        for (_, last_sg) in last_beam:
            if not getattr(last_sg, 'visited', False):
                setattr(last_sg, 'visited', True)
                for sel in tqdm(selectors):
                    # create a clone
                    new_selectors = list(last_sg.selectors)
                    if sel not in new_selectors:
                        new_selectors.append(sel)
                        sg = Conjuction(new_selectors)
                        sg_vector = sg.covers(data)
                        outcome_vector = target.covers (data)
                        quality = computeScore(sg_vector, outcome_vector, measure)
                        add_if_required(beam, sg, quality, beam_width, check_for_duplicates=True)
        depth += 1

    result = beam[:result_set_size]
    result.sort(key=lambda x: x[0], reverse=True)
    return result






# def 
# def main():
#     data=readData()
#     target=createTarget("outcome",True)
#     selectors = createSelectors(data,["outcome"])
#     with open("result.txt","w") as f:
#         for measure in ["accuracy", "oddsRatio", "colligation", "goodman", "f1"]:
#             f.write(measure)
#             f.write("\n")
#             result = simpleSearch(target, selectors, data, measure)
#             for r in result:
#                 f.write("\t"+str(r))
#                 f.write("\n")
#             f.write("\n")
#     print("end finished")
#     return result


def main():
    data=readData()
    target=createTarget("outcome",True)
    selectors = createSelectors(data,["outcome"])
    with open("result.txt","w") as f:
        for measure in ["accuracy", "oddsRatio", "colligation", "goodman", "f1"]:
            f.write(measure)
            f.write("\n")
            result = simpleSearch(target, selectors, data, measure)
            for r in result:
                f.write("\t"+str(r))
                f.write("\n")
            f.write("\n")
    print("end finished")
    return result


def main_beam():
    data=readData()
    target=createTarget("outcome",True)
    selectors = createSelectors(data,["outcome"])
    with open("result_beam.txt","w") as f:
        for measure in ["colligation"]:
            f.write(measure)
            f.write("\n")
            result = beamSearch(target, selectors, data, measure)
            for r in result:
                f.write("\t"+str(r))
                f.write("\n")
            f.write("\n")
    print("end finished")
    return result


def pruneFeatures(X, Y, feat_score, ignore, threshold):
    to_be_pruned = []
    for attr_name in [x for x in X if x not in ignore]:
        if feat_score[int(attr_name)].item()<threshold:
            to_be_pruned.append(attr_name)
    return to_be_pruned    
    

def L1_greedy(V,target, X, Y, measure, beam_width):
    to_be_pruned = pruneFeatures(X, Y, V, [], 0.4)
    [selectors, original_features, sg_to_index] = createSelectors(X, to_be_pruned)    
    scores= np.zeros((len(selectors)))
    # beam = [(0, Conjuction([]))]
    last_beam = [(0, Conjuction([]))]
    for index, sel in enumerate(tqdm(selectors)):
        sg = Conjuction([sel])
        sg_vector = sg.covers(X)
        outcome_vector = target.covers (Y)
        quality = computeQuality(sg_vector, outcome_vector, measure)
        scores[index] = quality
        add_if_required(last_beam, sg, quality, beam_width, check_for_duplicates=True)
    last_beam.sort(key=lambda x: x[0], reverse=True)
    return [last_beam, scores, selectors, original_features, sg_to_index]



def beamSearch_auxData(V, W, target, X,Y, measure, max_depth=2, beam_width=4, result_set_size=4):
    n_0=1
    last_beam = None

    F= np.zeros(W.shape)
    [beam, Q, selectors, original_features, sg_to_index] = L1_greedy(V,target, X, Y, measure, beam_width)

    new_W= np.zeros((len(selectors),len(selectors)))
    for i in range(len(new_W)):
        for j in range(len(new_W)):
            new_W[i,j] = W[original_features[i], original_features[j]] 

    depth = 0
    while beam != last_beam and depth < max_depth:
        last_beam = beam.copy()
        print("last_beam size: {}, depth: {}".format(len(last_beam), depth))
        for i in range(beam_width):
            (i_score, last_sg) = last_beam[i]
            if not getattr(last_sg, 'visited', False):
                setattr(last_sg, 'visited', True)
                FHat = np.zeros(len(selectors))
                for j in tqdm(range(len(selectors))):
                    
                    sel= selectors[j]                                                                                                                                            
                    new_selectors = list(last_sg.selectors)
                    sg = Conjuction(new_selectors)
                    sg_vector = sg.covers(X)
                    n = np.sum(sg_vector)

                    if n>n_0 and sel not in new_selectors:
                        FHat[j] = np.dot(new_W[j], Q+i_score)
                    else:
                        FHat[j] = 0
                j= np.argmax(FHat)    
                sg_vector = selectors[j].covers(X)
                outcome_vector = target.covers (Y)
                quality = computeQuality(sg_vector, outcome_vector, measure)
                add_if_required(beam, sg, quality, beam_width, check_for_duplicates=True)
        depth += 1
    result = beam[:result_set_size]
    result.sort(key=lambda x: x[0], reverse=True)
    return result


def main_beam_auxData():
    [X, Y, V, W] = readData()
    target = createTarget("outcome",True)
    # to_be_pruned = pruneFeatures(data, V, "outcome", 0.4)
    # selectors = createSelectors(data,["outcome"]+to_be_pruned)
    with open("result_beam.txt","w") as f:
        for measure in ["colligation"]:
            f.write(measure)
            f.write("\n")
            result = beamSearch_auxData(V,W,target, X, Y, measure)
            for r in result:
                f.write("\t"+str(r))
                f.write("\n")
            f.write("\n")
    print("end finished")
    return result


def test(ar):
    ar[2]=1
    ar[0]=1000

