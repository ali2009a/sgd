import pandas as pd
from itertools import combinations, chain
from math import factorial
from tqdm import tqdm
import numpy as np
from heapq import heappush, heappop


def readData():
    features = pd.read_csv("data/features.csv")
    features = features.iloc[:,:-5]
    clinicalData  =pd.read_excel("data/ClinicalData.xlsx")
    features["outcome"] = clinicalData["Cardiotoxicity"]
    features=features.set_index("index")
    return features


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
    for attr_name in [x for x in data if x not in ignore]:
        for val in pd.unique(data[attr_name]):
            selector = EquitySelector(attr_name, val)
            selectors.append(selector)
    return selectors


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
        self.selectors=selectors

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




def add_if_required(result, sg, quality, result_set_size):
    # if quality > task.min_quality:
    #     if check_for_duplicates and (quality, sg) in result:
    #         return
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


    n11 = tab.loc[1][1]
    n10 = tab.loc[1][0]
    n01 = tab.loc[0][1]
    n00 = tab.loc[0][0]

    if measure=="accuracy":
        quality = (n11+n00)/n
    elif measure=="oddsRatio":
        quality = (n00*n11)/(n10*n01)
    elif measure=="colligation":
        quality= ( n11*n00 - n10*n01 )/( n11*n00 + n10*n01 )
    elif measure=="goodman":
        quality = 1- ((min(n11,n10)+min(n00,n01))/(min(n01,n10)))
    elif measure=="f1":
        quality = (2*n11)/(n10+n01)

    return quality

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
    print ("simple search finished")
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
