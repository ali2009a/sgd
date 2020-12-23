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



# def readData():
#     data_root= "data3_toy"

#     # header=  list(range(0,100))
#     header=  list(range(0,4))
#     header2 = [str(x) for x in header]
#     features = pd.read_csv("{}/X.csv".format(data_root), header=None , names=header2)
#     outcome = pd.read_csv("{}/Y.csv".format(data_root), header=None, names=["outcome"])
#     result = pd.concat([features, outcome], axis=1, sort=False)

#     feat_score= pd.read_csv("{}/S1.csv".format(data_root), header=None)
#     feat_similarity= pd.read_csv("{}/S2.csv".format(data_root), header=None)
#     # feat_similarity = 1-feat_dissimilarity
#     feat_similarity = np.array(feat_similarity)
#     # features = features.iloc[:,:-5]
#     # clinicalData  =pd.read_excel("data/ClinicalData.xlsx")
#     # features["outcome"] = clinicalData["Cardiotoxicity"]
#     # features=features.set_index("index")
#     X=result.iloc[:,:-1]
#     Y=result.iloc[:,-1:]
#     return [X, Y, feat_score, feat_similarity]


def getNumberOfFeatuers(data_root):
    features = pd.read_csv("{}/X.csv".format(data_root), header=None)
    return features.shape[1]

def readData():
    print("reading data...")
    data_root= "sim_data_dec22"
    features_num = getNumberOfFeatuers(data_root)
    # header=  list(range(0,100))
    header=  list(range(0,features_num))
    header2 = [str(x) for x in header]
    features = pd.read_csv("{}/X.csv".format(data_root), header=None , names=header2)
    outcome = pd.read_csv("{}/Y.csv".format(data_root), header=None, names=["outcome"])
    result = pd.concat([features, outcome], axis=1, sort=False)

    feat_score= pd.read_csv("{}/V.csv".format(data_root), header=None)
    feat_similarity= pd.read_csv("{}/W.csv".format(data_root), header=None)
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

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))

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
        for val in  np.sort(pd.unique(data[attr_name])):
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

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))

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
    if check_for_duplicates and (quality, sg) in result:
        print("duplicated found")
        return

    sg_set = convertSGtoSet(sg)
    for pair in result:
        beam_quality = pair[0]
        sg_beam = pair[1]
        sg_beam_set = convertSGtoSet(sg_beam)
        subtract =  sg_beam_set - sg_set
        if len(subtract)==0 and quality < beam_quality:
            # print("Found a subset with better score! Not added")
            return

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
    # print(tab)
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
    # print("{}*{} - {}*{}".format(n11,n00,n10,n01))
    # print("{}*{} - {}*{}".format(n1b,n0b,nb1,nb0))
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
    

def L1_greedy(V,target, X, Y, measure, beam_width, threshold):
    computedScores={}
    to_be_pruned = pruneFeatures(X, Y, V, [], threshold)
    [selectors, original_features, sg_to_index] = createSelectors(X, to_be_pruned)    
    last_beam = [(0, Conjuction([]))]
    for index, sel in enumerate(tqdm(selectors)):
        sg = Conjuction([sel])
        sg_vector = sg.covers(X)
        outcome_vector = target.covers (Y)
        quality = computeQuality(sg_vector, outcome_vector, measure)
        add_if_required(last_beam, sg, quality, beam_width, check_for_duplicates=True)
        computedScores[sg]=quality
    return [last_beam, computedScores]






def convertSGtoSet(sg):
    res =set()
    for sel in sg.selectors:
        res.add(sel)
    return res


def initalizeScoreMatrix(F, computedScores, sg_to_beamIndex, sg_to_index, selectors, visited):
    print ("initalize Score Matrix")
    scoresSum=0
    for key , score in computedScores.items():
        scoresSum = scoresSum+score
    mean = scoresSum/len(computedScores)
    F[:,:] = mean
    for beam_key in sg_to_beamIndex:
        for sel in selectors:
            new_sg=Conjuction(beam_key.selectors + [sel])
            if new_sg in computedScores:
                F[sg_to_beamIndex[beam_key], sg_to_index[sel]] = computedScores[new_sg]
                visited[sg_to_beamIndex[beam_key], sg_to_index[sel]] = 1

def updateScoreMatrix(F, computedScores, sg_to_beamIndex, sg_to_index, selectors, visited):
    print ("Update Score Matrix")
    for beam_key in sg_to_beamIndex:
        for sel in selectors:
            new_sg=Conjuction(beam_key.selectors + [sel])
            if new_sg in computedScores:
                F[sg_to_beamIndex[beam_key], sg_to_index[sel]] = computedScores[new_sg]
                visited[sg_to_beamIndex[beam_key], sg_to_index[sel]] = 1



def createNewWeightMatrix(selectors, original_features, W):
    print("create new weight matrix...")
    new_W= np.zeros((len(selectors),len(selectors)))
    for i in tqdm(range(len(new_W))):
        for j in range(len(new_W)):
            new_W[i,j] = W[original_features[i], original_features[j]] 
    return new_W

def initializeVisitedMatrix(last_beam, selectors, sg_to_beamIndex, sg_to_index):
    visited= np.zeros((len(last_beam),len(selectors))) 
    for beam_key in sg_to_beamIndex:
        beam_key_set = convertSGtoSet(beam_key)
        for sel in beam_key_set:
            visited[sg_to_beamIndex[beam_key], sg_to_index[sel]] = 1
    return visited

def create_sg_to_beamIndex(last_beam):
    sg_to_beamIndex ={}
    for i, (i_score, last_sg) in enumerate(last_beam):
        sg_to_beamIndex[last_sg] = i
    return sg_to_beamIndex


def findPair(i, F, new_W, visited, last_sg, sg_to_beamIndex):
    j_prime = np.argmax(F[i,:])
    weights_vector=new_W[:, j_prime]
    weights_vector = weights_vector * (1-visited[sg_to_beamIndex[last_sg]]) 
    j = np.argmax(weights_vector)
    return [j, j_prime]

def create_sg_vector(selectors, last_sg, j, X):
    sel= selectors[j]                                                                                                                                            
    new_selectors = last_sg.selectors+[sel]
    sg = Conjuction(new_selectors)
    sg_vector = sg.covers(X)
    return sg, sg_vector

def printHistory(history, sg):
    for pair in history[sg]:
        print("{}               {}".format(pair[0], pair[1]))


def track_history(history, sg, sel):
    if sg in history:
        history[sg].append(sel)
    else:
        history[sg] = [sel]

def beamSearch_auxData_greedy(V, W, target, X,Y, measure, max_depth=2, beam_width=10, result_set_size=10, threshold=0.3, min_support=1, u=70):
    last_beam = None     
    depth = 0
    attemps_threshold=1000
    F= np.zeros(W.shape)
    [beam, computedScores] = L1_greedy(V,target, X, Y, measure, beam_width, threshold)
    print(beam)
    [selectors, original_features, sg_to_index] = createSelectors(X, []) 
    new_W = createNewWeightMatrix(selectors, original_features, W)

    history ={}

    while beam != last_beam and depth < max_depth-1:
        print("depth:{}".format(depth+2))
        last_beam = beam.copy()
        F= np.zeros((beam_width,len(selectors)))
        sg_to_beamIndex = create_sg_to_beamIndex(last_beam)
        visited = initializeVisitedMatrix(last_beam, selectors, sg_to_beamIndex, sg_to_index)
        initalizeScoreMatrix(F, computedScores, sg_to_beamIndex, sg_to_index, selectors, visited)
        for i in range(beam_width-1, -1,-1):
            print("expanding {}".format(last_beam[i][1]))
            (i_score, last_sg) = last_beam[i]
            if not getattr(last_sg, 'visited', False):
                setattr(last_sg, 'visited', True)
                pair_count=0
                attemps=0
                while pair_count<u and attemps<attemps_threshold:
                    attemps=attemps+1
                    j, j_prime = findPair(i, F, new_W, visited, last_sg, sg_to_beamIndex)
                    sg, sg_vector = create_sg_vector(selectors, last_sg, j, X)
                    n = np.sum(sg_vector)
                    if n<min_support: 
                        visited[i, j]=1                     
                        continue
                    outcome_vector = target.covers(Y)           
                    quality = computeQuality(sg_vector, outcome_vector, measure)
                    F[i,j] = quality
                    visited[i, j]=1
                    track_history(history, last_sg, (selectors[j],selectors[j_prime]))
                    add_if_required(beam, sg, quality, beam_width, check_for_duplicates=True)
                    computedScores[sg] = quality
                    pair_count= pair_count+1
            updateScoreMatrix(F, computedScores, sg_to_beamIndex, sg_to_index, selectors, visited)
        depth += 1
    result = beam[:result_set_size]
    result.sort(key=lambda x: x[0], reverse=True)
    return [result , computedScores, history]



def main_beam_auxData():
    [X, Y, V, W] = readData()
    target = createTarget("outcome",True)
    # to_be_pruned = pruneFeatures(data, V, "outcome", 0.4)
    # selectors = createSelectors(data,["outcome"]+to_be_pruned)
    with open("result_beam.txt","w") as f:
        for measure in ["colligation"]:
            f.write(measure)
            f.write("\n")
            result = beamSearch_auxData(V,W,target, X, Y, measure, max_depth=2, beam_width=100, result_set_size=100,threshold=0.54)
            for r in result:
                f.write("\t"+str(r))
                f.write("\n")
            f.write("\n")
    print("end finished")
    return result

def main_beam_auxData_greedy():
    [X, Y, V, W] = readData()
    target = createTarget("outcome",True)
    # to_be_pruned = pruneFeatures(data, V, "outcome", 0.4)
    # selectors = createSelectors(data,["outcome"]+to_be_pruned)
    with open("result_beam.txt","w") as f:
        for measure in ["colligation"]:
            f.write(measure)
            f.write("\n")
            result = beamSearch_auxData_greedy(V,W,target, X, Y, measure, max_depth=2, beam_width=100, result_set_size=100,threshold=0.54)
            for r in result:
                f.write("\t"+str(r))
                f.write("\n")
            f.write("\n")
    print("end finished")
    return result


def test(ar):
    ar[2]=1
    ar[0]=1000

def printResult(res):
    for index, pair in enumerate(res[0]):
        print("{}, {}".format(pair[1], pair[0]))

def verifyTrueCause(beam, true_sg):
    for pair in beam:
        found_sg=pair[1]
        if true_sg== found_sg:
            return True
    return False


def test(selectors_input):
    [X, Y, V, W] = readData()
    target = createTarget("outcome",True)
    to_be_pruned = pruneFeatures(X, Y, V, [], 0)
    [selectors, original_features, sg_to_index] = createSelectors(X, to_be_pruned)
    sg = Conjuction(selectors_input)
    sg_vector = sg.covers(X)
    outcome_vector = target.covers (Y)
    quality = computeQuality(sg_vector, outcome_vector, "")
    print (quality)

def get_score_sg(selectors_input, X, Y, V, W):
    target = createTarget("outcome",True)
    to_be_pruned = pruneFeatures(X, Y, V, [], 0)
    [selectors, original_features, sg_to_index] = createSelectors(X, to_be_pruned)
    sg = Conjuction(selectors_input)
    sg_vector = sg.covers(X)
    outcome_vector = target.covers (Y)
    quality = computeQuality(sg_vector, outcome_vector, "")
    print (quality)

# X1 X2 Rank Score 
# 48 48 0.000 0.659 
# 39 58 0.008 0.299 
# 44 66 0.002 0.365