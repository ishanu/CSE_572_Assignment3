import pandas as pd
import numpy as np
import math
import datetime
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from feature import getFeatures


def getCgmInsulinValues():

    CGMDataDf = pd.read_csv('CGMData.csv', usecols=["Date", "Time", "Sensor Glucose (mg/dL)"])

    CGMDataDf["Date"] = CGMDataDf["Date"].astype(str)
    CGMDataDf["Time"] = CGMDataDf["Time"].astype(str)
    CGMDataDf["DateTime"] = CGMDataDf["Date"] + " " + CGMDataDf["Time"]
    CGMDataDf.drop(columns=["Date", "Time"], inplace=True)
    CGMDataDf["DateTime"] = pd.to_datetime(CGMDataDf["DateTime"])

    # load the Insulin data

    InsulinDataDf = pd.read_csv('InsulinData.csv', usecols=["Date", "Time", "BWZ Carb Input (grams)"])

    # datetime formatting
    InsulinDataDf["Date"] = InsulinDataDf["Date"].astype(str)
    InsulinDataDf["Time"] = InsulinDataDf["Time"].astype(str)
    InsulinDataDf["DateTime"] = InsulinDataDf["Date"] + " " + InsulinDataDf["Time"]
    InsulinDataDf.drop(columns=["Date", "Time"], inplace=True)
    InsulinDataDf["DateTime"] = pd.to_datetime(InsulinDataDf["DateTime"])

    return CGMDataDf, InsulinDataDf


CGMDataDf, InsulinDf = getCgmInsulinValues()


def decide_bin(x, vmin, total_bins):
    a = float((x - vmin) / 20)
    f = math.floor(a)
    if f == total_bins:
        f = f - 1
    return f


def getMeals(InsulinDataDF):
    # get date time of all meals intake
    filt = InsulinDataDF['BWZ Carb Input (grams)'].notnull() & InsulinDataDF['BWZ Carb Input (grams)'] != 0
    InsulinmealDf = InsulinDataDF.loc[filt][["DateTime", 'BWZ Carb Input (grams)']]
    InsulinmealDf = InsulinmealDf.sort_values(by="DateTime")
    minCarbInput = InsulinmealDf["BWZ Carb Input (grams)"].min()
    maxCarbInput = InsulinmealDf["BWZ Carb Input (grams)"].max()
    bins = math.ceil((maxCarbInput - minCarbInput) / 20)
    InsulinmealDf["bin"] = InsulinmealDf["BWZ Carb Input (grams)"].apply(
        lambda x: decide_bin(x, minCarbInput, bins))
    return InsulinmealDf, bins


InsulinMealDf, nP1 = getMeals(InsulinDf)


def filter_meals(Insulin_all_meal_DF):
    # filter meal times by checking if another meal is not happening within 2hrs
    filt = []
    mintd = 2 * 60 * 60
    for i in range(len(Insulin_all_meal_DF) - 1):
        td = Insulin_all_meal_DF.iloc[i + 1]["DateTime"] - Insulin_all_meal_DF.iloc[i]["DateTime"]
        if td.total_seconds() <= mintd:
            filt.append(False)
        else:
            filt.append(True)
    filt.append(True)
    Insulin_meal_DF = Insulin_all_meal_DF[filt]
    return Insulin_meal_DF


InsulinData_P1_meal_DF = filter_meals(InsulinMealDf)


def extract_meal_CGM_data(Insulin_meal_DF, CGMData_DF):
    # extracting meal data from CGM data
    thirtymins = 30 * 60
    twohrs = 2 * 60 * 60
    colnames = list(range(1, 31))
    mealData_DF = pd.DataFrame()
    mealData_bins = []
    for i in range(len(Insulin_meal_DF)):
        lb = Insulin_meal_DF.iloc[i]["DateTime"] - datetime.timedelta(seconds=thirtymins)
        ub = Insulin_meal_DF.iloc[i]["DateTime"] + datetime.timedelta(seconds=twohrs)
        meal_bin = Insulin_meal_DF.iloc[i]["bin"]
        filt = (CGMData_DF["DateTime"] >= lb) & (CGMData_DF["DateTime"] < ub)
        filCGMData_DF = CGMData_DF[filt]
        if len(filCGMData_DF.index) == 30 and filCGMData_DF.isnull().values.any() == False:
            filCGMData_DF = filCGMData_DF.sort_values(by="DateTime")
            filCGMData_DF = filCGMData_DF.T
            filCGMData_DF.drop('DateTime', inplace=True)
            filCGMData_DF.reset_index(drop=True, inplace=True)
            filCGMData_DF.columns = colnames
            mealData_DF = mealData_DF.append(filCGMData_DF, ignore_index=True)
            mealData_bins.append(meal_bin)
    mealData_DF = mealData_DF.apply(pd.to_numeric)
    return mealData_DF, np.array(mealData_bins)


mealData_P1_DF, mealData_P1_bins = extract_meal_CGM_data(InsulinData_P1_meal_DF, CGMDataDf)


# visualization
# filCGMData_DF.plot(x="DateTime",y="Sensor Glucose (mg/dL)")
# mealData_P1_DF.plot()
# mealData_P1_DF.loc[0].plot()
# plt.show()

def extract_data(inp_DF):
    out_DF = getFeatures(inp_DF)
    return out_DF


mealData_ext_P1_DF = extract_data(mealData_P1_DF)
mealData_ext_P1_NP = mealData_ext_P1_DF.to_numpy()

# kmeans
kmeans_P1 = KMeans(n_clusters=int(nP1), random_state=0).fit(mealData_ext_P1_NP)

# dbscan
dbscan_P1 = DBSCAN(eps=50, min_samples=5, p=2).fit(mealData_ext_P1_NP)


def get_dbscan_means(labels, data):
    means_dbscan = []
    for i in range(max(labels) + 1):
        means_dbscan.append(data[labels == i].mean(axis=0))
    means_dbscan = np.array(means_dbscan)
    return means_dbscan


def change_minus_ones(labels, data, means_dbscan):
    dbl = np.copy(labels)
    # change all -1 to nearest cluster label
    for i in range(len(dbl)):
        if dbl[i] == -1:
            mindist = np.inf
            nl = None
            for j in range(max(dbl) + 1):
                d = np.sum((means_dbscan[j] - data[i]) ** 2)
                if d < mindist:
                    mindist = d
                    nl = j
            dbl[i] = j
    return dbl


def get_sse(labels, label, data):
    a = data[labels == label] - data[labels == label].mean(axis=0)
    return np.sum(a ** 2)


def get_max_sse_label(labels, data):
    maxsse = 0
    mssel = None
    for i in range(max(labels) + 1):
        a = get_sse(labels, i, data)
        if maxsse < a:
            maxsse = a
            mssel = i
    return mssel


def get_total_sse(labels, data):
    sumt = 0
    for i in range(max(labels) + 1):
        a = data[labels == i] - data[labels == i].mean(axis=0)
        sumt = np.sum(a ** 2)
    return sumt


def get_smallest_cluster_label(labels):
    minl = np.inf
    ml = None
    for i in range(max(labels) + 1):
        if np.sum(labels == i) < minl:
            minl = np.sum(labels == i)
            ml = i
    return ml


def divide_dbscan_labels(labels, bins, data):
    if max(labels) + 1 < bins:
        while max(labels) + 1 < bins:
            maxl = get_max_sse_label(labels, data)
            filt = labels == maxl
            nl = max(labels) + 1
            nd = data[filt]
            kmeans_nd = KMeans(n_clusters=2, random_state=0).fit(nd)
            kl = np.copy(kmeans_nd.labels_)
            kl[kl == 0] = maxl
            kl[kl == 1] = nl
            labels[filt] = kl
    return labels


def combine_dbscan_labels(labels, bins, data):
    if max(labels) + 1 > bins:
        while max(labels) + 1 > bins:
            minl = get_smallest_cluster_label(labels)
            maxl = max(labels)
            labels[labels == minl] = -1
            for i in range(maxl):
                if np.sum(labels == i) == 0:
                    labels[labels == maxl] = i
                    break
            md = get_dbscan_means(labels, data)
            labels = change_minus_ones(labels, data, md)
    return labels


# dbscan get N clusters
dbscan_labels_P1 = np.copy(dbscan_P1.labels_)
dbscan_labels_P1 = change_minus_ones(dbscan_labels_P1, mealData_ext_P1_NP,
                                     get_dbscan_means(dbscan_labels_P1, mealData_ext_P1_NP))
dbscan_labels_P1 = divide_dbscan_labels(dbscan_labels_P1, nP1, mealData_ext_P1_NP)
dbscan_labels_P1 = combine_dbscan_labels(dbscan_labels_P1, nP1, mealData_ext_P1_NP)

# kmeans sse
sse_kmeans_P1 = kmeans_P1.inertia_

# dbscan sse
sse_dbscan_P1 = get_total_sse(dbscan_labels_P1, mealData_ext_P1_NP)


# make the ground truth table
def make_gtt(gtlabels, labels, total_bins):
    gtt = np.zeros((total_bins, total_bins))
    for i in range(total_bins):
        a = gtlabels[labels == i]
        for j in range(len(a)):
            gtt[i][int(a[j])] = gtt[i][int(a[j])] + 1
    return gtt


gtt_kmeans_P1 = make_gtt(mealData_P1_bins, kmeans_P1.labels_, int(nP1))
gtt_dbscan_P1 = make_gtt(mealData_P1_bins, dbscan_labels_P1, int(nP1))


# cal entropy
def cal_entropy(gtt):
    total = gtt.sum()
    total_bins = gtt.shape[0]
    total_entropy = 0
    for i in range(total_bins):
        cluster_sum = gtt[i].sum()
        if cluster_sum == 0:
            continue
        cluster_entropy = 0
        for j in range(total_bins):
            if gtt[i, j] == 0:
                continue
            a = gtt[i, j] / cluster_sum
            e = -a * np.log2(a)
            cluster_entropy = cluster_entropy + e
        total_entropy = total_entropy + (cluster_sum / total) * cluster_entropy
    return total_entropy


entropy_kmeans_P1 = cal_entropy(gtt_kmeans_P1)
entropy_dbscan_P1 = cal_entropy(gtt_dbscan_P1)


# cal purity
def cal_purity(gtt):
    total = gtt.sum()
    total_bins = gtt.shape[0]
    total_purity = 0
    for i in range(total_bins):
        cluster_max = gtt[i].max()
        cluster_sum = gtt[i].sum()
        if cluster_sum == 0:
            continue
        cluster_purity = cluster_max / cluster_sum
        total_purity = total_purity + (cluster_sum / total) * cluster_purity
    return total_purity


purity_kmeans_P1 = cal_purity(gtt_kmeans_P1)
purity_dbscan_P1 = cal_purity(gtt_dbscan_P1)

# print the result: kmeans sse, dbscan sse, kmeans entropy, dbscan entropy, kmeans purity, dbscan purity
# print(f'{sse_kmeans_P1} {sse_dbscan_P1} {entropy_kmeans_P1} {entropy_dbscan_P1} {purity_kmeans_P1} {purity_dbscan_P1}')

# save the results in a file
results = np.array(
    [[sse_kmeans_P1, sse_dbscan_P1, entropy_kmeans_P1, entropy_dbscan_P1, purity_kmeans_P1, purity_dbscan_P1]])
np.savetxt("Results.csv", results, delimiter=",", fmt="%10.4f")
