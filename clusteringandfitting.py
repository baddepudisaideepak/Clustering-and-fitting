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


cm = matplotlib.colormaps["Paired"]

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

print(worldMortality)
print(worldPoverty)


world = pd.concat([worldMortality, worldPoverty], axis=1)
world = world.head(-3)

print(world.head(3))




# create a scaler object
scaler = pp.RobustScaler()

# extract columns
df_clust = world[["Mortality", "Poverty"]]
# and set up the scaler
scaler.fit(df_clust)

# apply the scaling
df_norm = scaler.transform(df_clust)


# calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(df_norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")   # allow for minus signs

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=2, n_init=20)

# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm)     # fit done on x,y pairs

# extract cluster labels
labels = kmeans.labels_

# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

# extract x and y values of data points
x = world["Mortality"]
y = world["Poverty"]

plt.figure(figsize=(8.0, 8.0))

# plot data with kmeans cluster number
plt.scatter(x, y, 10, labels, marker="o", cmap=cm)

# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")

plt.xlabel("Mortality")
plt.ylabel("Poverty")
plt.title("Mortality vs Poverty")
plt.savefig("Mortality_Povery.png", dpi =300)
plt.show()
