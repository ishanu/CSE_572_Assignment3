import pandas as pd
import numpy as np
import math
import datetime
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize


# Please run this program on the data given in the second assignment ie files: 'CGMData670GPatient3.xlsx', 'InsulinAndMealIntake670GPatient3.xlsx'

def get_CGM_Insulin_DF(CGM_file, Insulin_file, isCSVdata):
    # load the CGM data

    CGMData_DF = pd.read_csv(CGM_file, usecols=["Date", "Time", "Sensor Glucose (mg/dL)"])


    # datetime formatting
    CGMData_DF["Date"] = CGMData_DF["Date"].astype(str)
    CGMData_DF["Time"] = CGMData_DF["Time"].astype(str)
    CGMData_DF["DateTime"] = CGMData_DF["Date"] + " " + CGMData_DF["Time"]
    CGMData_DF.drop(columns=["Date", "Time"], inplace=True)
    CGMData_DF["DateTime"] = pd.to_datetime(CGMData_DF["DateTime"])

    # load the Insulin data

    InsulinData_DF = pd.read_csv(Insulin_file, usecols=["Date", "Time", "BWZ Carb Input (grams)"])


    # datetime formatting
    InsulinData_DF["Date"] = InsulinData_DF["Date"].astype(str)
    InsulinData_DF["Time"] = InsulinData_DF["Time"].astype(str)
    InsulinData_DF["DateTime"] = InsulinData_DF["Date"] + " " + InsulinData_DF["Time"]
    InsulinData_DF.drop(columns=["Date", "Time"], inplace=True)
    InsulinData_DF["DateTime"] = pd.to_datetime(InsulinData_DF["DateTime"])

    return CGMData_DF, InsulinData_DF


#CGMData_P1_DF, InsulinData_P1_DF = get_CGM_Insulin_DF('CGMData670GPatient3.xlsx','InsulinAndMealIntake670GPatient3.xlsx', False)


# CGMData_P2_DF, InsulinData_P2_DF = get_CGM_Insulin_DF('CGMData.csv', 'InsulinData.csv', True)
CGMData_P1_DF, InsulinData_P1_DF = get_CGM_Insulin_DF('CGMData.csv', 'InsulinData.csv', True)

def decide_bin(x, vmin, total_bins):
    a = float((x - vmin) / 20)
    f = math.floor(a)
    if f == total_bins:
        f = f - 1
    return f


def get_all_meal_DF(InsulinData_DF):
    # get date time of all meals intake
    filt = InsulinData_DF['BWZ Carb Input (grams)'].notnull() & InsulinData_DF['BWZ Carb Input (grams)'] != 0
    Insulin_all_meal_DF = InsulinData_DF.loc[filt][["DateTime", 'BWZ Carb Input (grams)']]
    Insulin_all_meal_DF = Insulin_all_meal_DF.sort_values(by="DateTime")
    minCarbInput = Insulin_all_meal_DF["BWZ Carb Input (grams)"].min()
    maxCarbInput = Insulin_all_meal_DF["BWZ Carb Input (grams)"].max()
    total_bins = math.ceil((maxCarbInput - minCarbInput) / 20)
    Insulin_all_meal_DF["bin"] = Insulin_all_meal_DF["BWZ Carb Input (grams)"].apply(
        lambda x: decide_bin(x, minCarbInput, total_bins))
    return Insulin_all_meal_DF, total_bins


InsulinData_P1_all_meal_DF, N_P1 = get_all_meal_DF(InsulinData_P1_DF)


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


InsulinData_P1_meal_DF = filter_meals(InsulinData_P1_all_meal_DF)


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


mealData_P1_DF, mealData_P1_bins = extract_meal_CGM_data(InsulinData_P1_meal_DF, CGMData_P1_DF)


# visualization
# filCGMData_DF.plot(x="DateTime",y="Sensor Glucose (mg/dL)")
# mealData_P1_DF.plot()
# mealData_P1_DF.loc[0].plot()
# plt.show()

# Feature extraction
# features:
# tmax - tm
# CGM_max - CGM_min
# max CGM velocity, time at which the velocity is max
# FTT - half sinusoidal - get two most dominant frequency buckets
# windowed mean - window size = 6: you will get 4 and 5 means
# take middle 5 means - window size = 3

def absorption_time(row):
    #     return 5*int(row.idxmax(skipna = True))
    if row.size == 30:
        newrow = row.iloc[6:30]
        return 5 * int(newrow.idxmax(skipna=True))
    else:
        return 5 * int(row.idxmax(skipna=True))


def CGM_max_velocity(row):
    vmax = None
    vmaxtime = None
    for i in range(row.size):
        if i == 0:
            v = (row.iloc[i + 1] - row.iloc[i]) / 5
        elif i == row.size - 1:
            v = (row.iloc[i] - row.iloc[i - 1]) / 5
        else:
            v = (row.iloc[i + 1] - row.iloc[i - 1]) / 10
        if vmax == None or v > vmax:
            vmax = v
            vmaxtime = i * 5
    return (vmax, vmaxtime)


def CGM_FFT(row):
    sp = np.fft.fft(row)
    power = np.square(sp.real) + np.square(sp.imag)
    freq = np.fft.fftfreq(row.size, d=300)
    mp = 0
    mp2 = 0
    mpi = None
    mp2i = None
    for i in range(1, row.size):
        p = power[i]
        f = freq[i]
        if p > mp:
            mp2 = mp
            mp2i = mpi
            mp = p
            mpi = f
        elif p > mp2:
            mp2 = p
            mp2i = f

    ip = 0
    ip2 = 0
    ipi = None
    ip2i = None
    for i in range(1, row.size):
        p = power[i]
        f = freq[i]
        if i < row.size - 1 and p > power[i - 1] and p > power[i + 1]:
            if p > ip:
                ip2 = ip
                ip2i = ipi
                ip = p
                ipi = f
            elif p > ip2:
                ip2 = p
                ip2i = f
    if ipi != None and ip2i != None:
        return (ipi, ip2i)

    return (mpi, mp2i)


# take mean of middle 5 windows of 3 length
def windowed_mean(row):
    if row.size == 30:
        newrow = row.iloc[7:22]
    else:
        newrow = row.iloc[4:19]
    avgs = []
    for i in range(5):
        m = (newrow.iloc[i * 3] + newrow.iloc[i * 3 + 1] + newrow.iloc[i * 3 + 2]) / 3
        avgs.append(m)
    return (avgs[0], avgs[1], avgs[2], avgs[3], avgs[4])


def extract_data(inp_DF):
    out_DF = pd.DataFrame()
    out_DF['absorption_time (mins)'] = inp_DF.apply(lambda row: absorption_time(row), axis=1)
    out_DF['CGM_range'] = inp_DF.apply(lambda row: row.max() - row.min(), axis=1)

    CGM_velocity_data = inp_DF.apply(lambda row: CGM_max_velocity(row), axis=1)
    CGM_max_vel, CGM_max_vel_time = list(zip(*CGM_velocity_data))
    out_DF['CGM_max_vel'] = CGM_max_vel
    out_DF['CGM_max_vel_time'] = CGM_max_vel_time

    CGM_fft_data = inp_DF.apply(lambda row: CGM_FFT(row), axis=1)
    CGM_max_freq, CGM_max2_freq = list(zip(*CGM_fft_data))
    out_DF['CGM_max_freq'] = CGM_max_freq
    out_DF['CGM_max2_freq'] = CGM_max2_freq

    CGM_wm_data = inp_DF.apply(lambda row: windowed_mean(row), axis=1)
    CGM_wm1, CGM_wm2, CGM_wm3, CGM_wm4, CGM_wm5 = list(zip(*CGM_wm_data))
    out_DF['CGM_wm1'] = CGM_wm1
    out_DF['CGM_wm2'] = CGM_wm2
    out_DF['CGM_wm3'] = CGM_wm3
    out_DF['CGM_wm4'] = CGM_wm4
    out_DF['CGM_wm5'] = CGM_wm5

    return out_DF


mealData_ext_P1_DF = extract_data(mealData_P1_DF)
mealData_ext_P1_NP = mealData_ext_P1_DF.to_numpy()

# kmeans
kmeans_P1 = KMeans(n_clusters=int(N_P1), random_state=0).fit(mealData_ext_P1_NP)

# dbscan
dbscan_P1 = DBSCAN(eps=50, min_samples=5, p=2).fit(mealData_ext_P1_NP)


# dbscan_P1 = DBSCAN(eps=0.05, min_samples=5, p=2).fit(normalize(mealData_ext_P1_NP, axis=0))

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
dbscan_labels_P1 = divide_dbscan_labels(dbscan_labels_P1, N_P1, mealData_ext_P1_NP)
dbscan_labels_P1 = combine_dbscan_labels(dbscan_labels_P1, N_P1, mealData_ext_P1_NP)

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


gtt_kmeans_P1 = make_gtt(mealData_P1_bins, kmeans_P1.labels_, int(N_P1))
gtt_dbscan_P1 = make_gtt(mealData_P1_bins, dbscan_labels_P1, int(N_P1))


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
#print(f'{sse_kmeans_P1} {sse_dbscan_P1} {entropy_kmeans_P1} {entropy_dbscan_P1} {purity_kmeans_P1} {purity_dbscan_P1}')

# save the results in a file
results = np.array(
    [[sse_kmeans_P1, sse_dbscan_P1, entropy_kmeans_P1, entropy_dbscan_P1, purity_kmeans_P1, purity_dbscan_P1]])
np.savetxt("Results.csv", results, delimiter=",", fmt="%10.4f")