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


"""
    This function calculates the silhouette score for a specified number of clusters.
    :param xy: Array of data points.
    :param n: Number of clusters.
    :return: Silhouette score.
    """


def one_silhoutte(xy, n):

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)     # fit done on x,y pairs

    labels = kmeans.labels_

    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))

    return score


"""
    Exponential growth model function.
    :param t: Time variable.
    :param n0: Initial scale factor.
    :param g: Growth rate.
    :return: Calculated exponential function value.
    """


def exponential(t, n0, g):

    # makes it easier to get a guess for initial parameters
    t = t - 1990

    f = n0 * np.exp(g * t)

    return f


cm = matplotlib.colormaps["Paired"]


# Load and preprocess data
mortalityRate = pd.read_excel("Mortalityrate.xlsx")
povertyRate = pd.read_excel("Poverty.xlsx")
mortalityRate.set_index(mortalityRate["Country Name"], inplace=True)
mortalityRate.drop(['Series Name',
                    'Series Code',
                    "Country Name",
                    'Country Code'],
                   axis=1,
                   inplace=True)
povertyRate.set_index(povertyRate["Country Name"], inplace=True)
povertyRate.drop(['Series Name', 'Series Code', "Country Name",
                 'Country Code'], axis=1, inplace=True)

# Transform and set the years as index
years = np.linspace(1990, 2022, 33).astype(int)
mortalityRate = mortalityRate.T
mortalityRate.set_index(years, inplace=True)
povertyRate = povertyRate.T
povertyRate.set_index(years, inplace=True)

print("\nmortalityRate:", mortalityRate.head(5))
print("\npovertyRate:", povertyRate.head(5))


worldMortality = mortalityRate["World"]
worldPoverty = povertyRate["World"]

worldMortality.name = "Mortality"
worldPoverty.name = "Poverty"

print("\n worldMortality:", worldMortality)
print("\n worldPoverty:", worldPoverty)


world = pd.concat([worldMortality, worldPoverty], axis=1)
world = world.head(-3)

print("\n", world.head(3))


# Normalization and Clustering
scaler = pp.RobustScaler()
df_clust = world[["Mortality", "Poverty"]]
scaler.fit(df_clust)
df_norm = scaler.transform(df_clust)


# Silhouette score calculation for cluster numbers 2 to 10
for ic in range(2, 11):
    score = one_silhoutte(df_norm, ic)
    # allow for minus signs
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")


# Applying K-Means clustering and visualizing results
kmeans = cluster.KMeans(n_clusters=2, n_init=20)
kmeans.fit(df_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
x = world["Mortality"]
y = world["Poverty"]
plt.figure(figsize=(8.0, 8.0))
plt.scatter(x, y, 10, labels, marker="o", cmap=cm)
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("Mortality")
plt.ylabel("Poverty")
plt.title("Mortality vs Poverty")
plt.savefig("Mortality_Povery.png", dpi=300)
plt.show()


# Curve fitting and forecasting
world = world.reset_index()
world = world.rename(columns={"index": "years"})
param, covar = opt.curve_fit(
    exponential, world["years"], world["Mortality"], p0=(
        100, -0.03))
print("Mortality rate:-")
print('\nparam value')
print('param value', *param)

imlib.reload(err)

# forecast for one year
forecast = exponential(2030, *param)
sigma = err.error_prop(2030, exponential, param, covar)
print("\nforecast and sigma values ")
print(f"{forecast: 6.3e} +/- {sigma: 6.3e}")

# plotting mortality rate
plt.figure()
plt.plot(world["years"], world["Mortality"], '--', label="world")
plt.xlabel("year")
plt.ylabel("Mortality rate(out of 1000)")
plt.legend()
plt.title("World's Mortality rate")
plt.savefig("Mortality.png", dpi=300)
plt.show()


# create array for forecasting
year = np.linspace(1985, 2025, 100)
forecast = exponential(year, *param)
sigma = err.error_prop(year, exponential, param, covar)
up = forecast + sigma
low = forecast - sigma


# Plotting extended forecast with confidence intervals
plt.figure()
plt.plot(world["years"], world["Mortality"], '--', label="world")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("Mortality rate(/1000)")
plt.legend()
plt.title("World's Mortality rate forecast ")
plt.savefig("Mortality rate fitting.png", dpi=300)
plt.show()

param, covar = opt.curve_fit(
    exponential, world["years"], world["Poverty"], p0=(
        100, -0.03))

print("Poverty rate:-")
print('\nparam value')
print(*param)

# forecast for one year
forecast = exponential(2030, *param)
sigma = err.error_prop(2030, exponential, param, covar)
print("\nforecast and sigma values ")
print(f"{forecast: 6.3e} +/- {sigma: 6.3e}")

# plotting poverty rate
plt.figure()
plt.plot(world["years"], world["Poverty"], '--', label="world")
plt.xlabel("year")
plt.ylabel("Poverty rate")
plt.legend()
plt.title("World's Poverty rate")
plt.savefig("Poverty.png", dpi=300)
plt.show()

# create array for forecasting
year = np.linspace(1985, 2025, 100)
forecast = exponential(year, *param)
sigma = err.error_prop(year, exponential, param, covar)
up = forecast + sigma
low = forecast - sigma


# Plotting extended forecast with confidence intervals
plt.figure()
plt.plot(world["years"], world["Poverty"], '--', label="world")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("Poverty rate")
plt.legend()
plt.title("World's Poverty rate forecast ")
plt.savefig("Poverty rate fitting.png", dpi=300)
plt.show()
