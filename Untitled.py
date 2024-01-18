import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import matplotlib
import scipy.optimize as opt
import importlib as imlib
import errors as err


""" Calculates silhoutte
     score for n clusters """

def one_silhoutte(xy, n):

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)     # fit done on x,y pairs

    labels = kmeans.labels_

    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))

    return score


"""Calculates exponential function
    with scale factor n0
    and growth rate g.  """
def exponential(t, n0, g):

    # makes it easier to get a guess for initial parameters
    t = t - 1990

    f = n0 * np.exp(g*t)

    return f


mortalityRate = pd.read_excel("Mortalityrate.xlsx")
povertyRate = pd.read_excel("Poverty.xlsx")


mortalityRate.set_index(mortalityRate["Country Name"], inplace= True)
mortalityRate.drop(['Series Name', 'Series Code',"Country Name",'Country Code'],axis=1,inplace = True)

povertyRate.set_index(povertyRate["Country Name"], inplace= True)
povertyRate.drop(['Series Name', 'Series Code',"Country Name",'Country Code'],axis=1,inplace = True)

years = np.linspace(1990,2022,33).astype(int)


mortalityRate = mortalityRate.T
mortalityRate.set_index(years,inplace = True)
povertyRate = povertyRate.T
povertyRate.set_index(years,inplace = True)

print(mortalityRate.head(5))
print(povertyRate.head(5))
worldMortality = mortalityRate["World"]
worldPoverty = povertyRate["World"]

worldMortality.name = "Mortality"
worldPoverty.name = "Poverty"

world = pd.concat([worldMortality, worldPoverty], axis=1)
world = world.head(-3)

print(world.head(3))
