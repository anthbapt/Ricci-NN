import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from keras.models import load_model
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform, pdist
import os
import glob
path = "/home/anthbapt/Documents/fMNIST_DNN_training/wk"
os.chdir(path)

# Load file and model
model = np.load("model_predict.npy",allow_pickle = True)
accuracy = np.load("accuracy.npy")
#remove number of features
x_test = np.array(pd.read_csv("x_test.csv", header = None))[1::]

k = 30
a = 0.98


def corr_analysis(scalar_curvs1, accuracy, x_test):
    # Generate FR curvature data frame
    accuracy = np.asarray(accuracy)
    acc = accuracy[::2]
    s = scalar_curvs1[np.where(acc > a)[0][0]]
    sr = s[0]
    for i in range(1, len(s)):
        sr = np.column_stack((sr, s[i]))
    for i in np.where(acc > a)[1:]:
        s = scalar_curvs1[i]
        sr2 = s[0]
    
        for j in range(1, len(s)):
            sr2 = np.column_stack((sr2, s[j]))
        sr = np.column_stack((sr, sr2))
    
    # Set column names
    column_names = np.repeat(np.arange(1, len(scalar_curvs1[0]) + 1), len(np.where(acc > a)[0]))
    pd.DataFrame(sr, columns=column_names)

    # Extract to data frame summary
    ssr = sr[:, 0].copy()
    for i in range(1, sr.shape[1]):
        ssr = np.column_stack((ssr, sr[:, i]))
    layer = np.repeat(1, x_test.shape[0])
    for i in range(2, len(scalar_curvs1[0])):
        layer = np.column_stack((layer, np.repeat(i, x_test.shape[0])))
    layer_all = np.repeat(layer, len(np.where(acc > a)[0]))
    mod = np.repeat(1, len(scalar_curvs1[0]) * x_test.shape[0])
    for i in range(2, len(np.where(acc > a)[0])):
        mod = np.column_stack((mod, np.repeat(i, len(scalar_curvs1[0]) * x_test.shape[0])))
    data = pd.DataFrame(np.column_stack((ssr, layer_all, mod)), columns=["ssr", "layer", "mod"])
    
    return data
    

def filter_(fr_data, sc_data):
    # Deal with infinite geodesics by removing models with these
    d = sc_data.groupby(['layer', 'mod'])['ssr'].agg(['mean', 'std']).reset_index()
    # Replace Inf values with NaN for further processing
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Remove rows with NaN in the mean column
    d.dropna(subset=['mean'], inplace=True)
    # Find models with Inf mean values
    rmod = d.loc[(d['mean'] == np.inf) | (d['mean'] == -np.inf), 'mod'].tolist()
    # Filter sc_data and fr_data based on rmod
    if rmod:
        m = sc_data['mod'].isin(rmod)
        rm2 = np.where(~m)[0]
        sc_data2 = sc_data.iloc[rm2].copy()
    else:
        sc_data2 = sc_data.copy()

    if rmod:
        m = fr_data['mod'].isin(rmod)
        rm2 = np.where(~m)[0]
        fr_data2 = fr_data.iloc[rm2].copy()
    else:
        fr_data2 = fr_data.copy()

    # Aggregate the data frames and sum over data points
    d = sc_data2.groupby(['layer', 'mod'])['ssr'].agg(['mean', 'std']).reset_index()
    d.columns = ['layer', 'mod', 'sd', 'mean']

    # Print maximum value excluding Inf
    max_val = d.loc[d['mean'].replace([np.inf, -np.inf], np.nan).idxmax(skipna=True)]
    print(max_val)
    
    # Aggregate the data frames and sum over data points
    msc = sc_data2.groupby(['layer', 'mod'])['ssr'].sum().reset_index()
    mfr = fr_data2.groupby(['layer', 'mod'])['ssr'].sum().reset_index()
    
    return mfr, msc
        
    
def create_figure(msc, mfr):
    plt.figure(figsize=(8, 8))

    # Geodesics over layer
    plt.subplot(2, 2, 1)
    msc.boxplot(column='ssr', by='layer', grid=False)
    plt.xlabel('Layer')
    plt.ylabel('Total geodesic change from prior layer')

    # FR curvature over layer
    plt.subplot(2, 2, 2)
    mfr.boxplot(column='ssr', by='layer', grid=False)
    plt.xlabel('Layer')
    plt.ylabel('Total FR Curvature')

    # Skip layer FR curvature vs geodesic
    plt.subplot(2, 2, 3)
    plt.scatter(msc.loc[msc['layer'] != 1, 'ssr'], mfr.loc[mfr['layer'] != mfr['layer'].max(), 'ssr'],
                c=mfr.loc[mfr['layer'] != mfr['layer'].max(), 'layer'], marker='o', cmap='viridis')
    plt.xlabel('Total geodesic change from prior layer from l-1->l')
    plt.ylabel('Total FR Curvature of l-1')
    plt.title('Layer Skip')
    plt.plot(np.unique(msc.loc[msc['layer'] != 1, 'ssr']),
             np.poly1d(np.polyfit(msc.loc[msc['layer'] != 1, 'ssr'], mfr.loc[mfr['layer'] != mfr['layer'].max(), 'ssr'], 1))
             (np.unique(msc.loc[msc['layer'] != 1, 'ssr'])), color='red')

    plt.tight_layout()
    plt.close()


# Set intermediate lists to loop through the models
scalar_curvs2 = np.empty(len(model), dtype = object)
FR1 = np.empty(len(model), dtype = object)

# Loop through models
for j in range(len(model)):
    activations = model[j]
    # Compute kNN graphs
    gs1 = np.empty(len(activations), dtype = object)
    for i in range(len(activations)):
        av = activations[i]
        neigh = NearestNeighbors(metric="euclidean", n_neighbors=k)
        neigh.fit(av)
        gs1[i] = neigh.kneighbors_graph(av)
        print(gs1[i])
        
    neigh.fit(x_test)
    g0 = neigh.kneighbors_graph(x_test)

    # Compute FR curvatures for each act # Assuming activations is a 0-based list in Python # Assuming activations is a 0-based list in Pythonivation
    F_n_list1 = []
    for i in range(len(gs1)):
        graph_i = nx.from_scipy_sparse_array(gs1[i])
        D = np.array([graph_i.degree(i) for i in graph_i.nodes])
        F_mat = 4 - np.outer(D, D)
        a_mat = nx.to_numpy_array(graph_i)
        F_mat[a_mat == 0] = 0
        F_n1 = np.sum(F_mat, axis=1)
        F_n_list1.append(F_n1)

    Ric1 = np.empty(len(gs1), dtype = object)
    Ric1[0] = np.subtract(squareform(pdist(gs1[0].todense())), squareform(pdist(g0.todense())))
    
    # Compute Ric1 for each i
    for i in range(1, len(gs1)):
        Ric1[i] = np.subtract(squareform(pdist(gs1[i].todense())), squareform(pdist(gs1[i-1].todense())))
    
    # Compute sum of Ric1 over data points
    sc2 = [np.apply_along_axis(lambda x: np.sum(x, axis=0, keepdims=True), 1, ric) for ric in Ric1]
    # Output
    scalar_curvs2[j] = sc2
    FR1[j] = F_n_list1
    
    
fr_data = corr_analysis(FR1, accuracy, x_test)
sc_data = corr_analysis(scalar_curvs2, accuracy, x_test)
mfr, msc = filter_(fr_data, sc_data)
create_figure(msc, mfr)


# Calculate correlations
aa = pearsonr(msc['ssr'], mfr['ssr'])
aa2 = pearsonr(msc.loc[msc['layer'] != 1, 'ssr'], mfr.loc[mfr['layer'] != mfr['layer'].max(), 'ssr'])

print("gdists: ", msc)
print("FR: ", mfr)
print("correlation: ", [aa[0], aa[1]])
print("correlation_shift: ", [aa2[0], aa2[1]])
#print("n_mods: ", sum(acc > a))
