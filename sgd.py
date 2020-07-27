import pandas as pd

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


class BinaryTarget():
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value



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



