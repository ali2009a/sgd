import pandas as pd
from itertools import combinations, chain
from math import factorial
from tqdm import tqdm
import numpy as np
from heapq import heappush, heappop


def readData():
    features = pd.read_csv("data/features.csv")
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
        print (attr_name)
        for val in pd.unique(data[attr_name]):
            print (val)
            selector = EquitySelector(attr_name, val)
            selectors.append(selector)
    print (selectors)
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



def add_if_required(result, sg, quality, result_set_size):
    # if quality > task.min_quality:
    #     if check_for_duplicates and (quality, sg) in result:
    #         return
    if len(result) < result_set_size:
        heappush(result, (quality, sg))
    elif quality > result[0][0]:
        heappop(result)
        print (result)
        print (quality)
        print (sg)
        heappush(result, (quality, sg))


def computeScore(sg_vector, outcome_vector):
    n=len(sg_vector)
    tab = pd.crosstab(sg_vector,outcome_vector)
    n11 = tab[1][1]
    n10 = tab[1][0]
    n01 = tab[0][1]
    n00 = tab[0][0]
    quality = (n11+n00)/n
    return quality

def simpleSearch(target, selectors, data):
    searchSpace = createSearchSpace(selectors,2)
    print (searchSpace)
    tqdm_searchSpace = tqdm(searchSpace[0],total=searchSpace[1])
    result = []
    for selectors_one_point in tqdm_searchSpace:
        sg = Conjuction(selectors_one_point)
        sg_vector = sg.covers(data)
        outcome_vector = target.covers(data)
        print(len(sg_vector))
        print(len(outcome_vector))
        quality = computeScore(sg_vector, outcome_vector)
        # result.append((quality,selectors_one_point))
        print (result)
        add_if_required(result, sg, quality, 3)
    return result


def main():
    data=readData()
    target=createTarget("outcome",True)
    selectors = createSelectors(data,["outcome"])
    simpleSearch(target, selectors, data)

