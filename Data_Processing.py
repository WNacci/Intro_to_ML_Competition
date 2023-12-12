import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#similarity function
def row_feature_similarity(row):
    pre = row["pre_feature_weights"]
    post = row["post_feature_weights"]
    return (pre * post).sum() / (np.linalg.norm(pre) * np.linalg.norm(post))

def row_feature_similarity_morph(row):
    pre = row["pre_morph_embeddings"]
    post = row["post_morph_embeddings"]
    if not isinstance(pre, np.ndarray):
        return 0
    return (pre * post).sum() / (np.linalg.norm(pre) * np.linalg.norm(post))
def row_feature_similarity_rfx(row):
    prex = row["pre_rf_x"]
    postx = row["post_rf_x"]
    return abs(prex-postx)
    
def row_feature_similarity_rfy(row):
    prey = row["pre_rf_y"]
    posty = row["post_rf_y"]
    return abs(prey-posty)

def row_feature_similarity(row):
    prey = row["pre_rf_y"]
    posty = row["post_rf_y"]
    prex = row["pre_rf_x"]
    postx = row["post_rf_x"]
    return np.sqrt((prey-posty)**2+(prex-postx)**2)

def nuclear_dist(row):
    prex = row["pre_nucleus_x"]
    postx = row["post_nucleus_x"]
    prey = row["pre_nucleus_y"]
    posty = row["post_nucleus_y"]
    prez = row["pre_nucleus_z"]
    postz = row["post_nucleus_z"]
    return np.sqrt((prey-posty)**2+(prex-postx)**2+(prez-postz)**2)

def oracle_diff(row):
    preo = row["pre_oracle"]
    posto = row["post_oracle"]
    return abs(preo-posto)

def total_path(row):
    pred = row["pre_skeletal_distance_to_soma"]
    postd = row["post_skeletal_distance_to_soma"]
    return abs(pred+postd)

def path_to_dist(row):
    return row["total_path"]/row["nuclear_dist"]

def same_clust(row):
    return row["pre_cluster"]==row["post_cluster"]

# Data prep function (Basic Feature Engineering)
def Basic_FE(data):
    # Add options here!
    data["fw_similarity"] = data.apply(row_feature_similarity, axis=1)
    data["morph_similarity"] = data.apply(row_feature_similarity_morph, axis=1)
    data["rf_x_similarity"] = data.apply(row_feature_similarity_rfx, axis=1)
    data["rf_y_similarity"] = data.apply(row_feature_similarity_rfy, axis=1)
    
    # Making new one-hot for compartment and projeciton group
    data["projection_group"] = (
        data["pre_brain_area"].astype(str)
        + "->"
        + data["post_brain_area"].astype(str)
    )
    data = pd.get_dummies(data, columns=['projection_group'],dtype=float)
    data = pd.get_dummies(data, columns=['compartment'],dtype=float)
    
    # Deleting unwanted features
    data = data.loc[:, ~data.columns.isin(['ID','pre_brain_area',
                                           'post_brain_area',
                                           'pre_feature_weights',
                                           'post_feature_weights',
                                           'pre_morph_embeddings',
                                           'post_morph_embeddings',
                                           'pre_nucleus_id',
                                           'post_nucleus_id',])]
    return data

from sklearn.cluster import SpectralClustering

#XGboost Preprocessing
def xg_FE(df):
    #Combine compartment?)
    
    df["fw_similarity"] = df.apply(row_feature_similarity, axis=1)
    df["morph_similarity"] = df.apply(row_feature_similarity_morph, axis=1)
    df["rf_similarity"] = df.apply(row_feature_similarity, axis=1)
    df["rf_x_similarity"] = df.apply(row_feature_similarity_rfx, axis=1)
    df["rf_y_similarity"] = df.apply(row_feature_similarity_rfy, axis=1)
    df["nuclear_dist"] = df.apply(nuclear_dist,axis=1)
    df["oracle_diff"] = df.apply(oracle_diff,axis=1)
    df["total_path"] = df.apply(total_path,axis=1)
    df["path_to_dist"] = df.apply(path_to_dist,axis=1)
    df["same_clust"] = df.apply(same_clust,axis=1)

    df.rename({'compartment': 'comp', 'pre_brain_area': 'poba','post_brain_area':'prba'}, axis=1, inplace=True)
    
    df["projection_g"] = (
        df["prba"].astype(str)
        + "->"
        + df["poba"].astype(str)
    )
    
    df["compartment"] = df["comp"].astype("category")
    df["projection_group"] = df["projection_g"].astype("category")
    #df["pre_brain_area"] = df["prba"].astype("category")
    #df["post_brain_area"] = df["poba"].astype("category")
    df.rename({'pre_cluster': 'pre_clust', 'post_cluster': 'post_clust'}, axis=1, inplace=True)
    df["pre_cluster"] = df["pre_clust"].astype("category")
    df["post_cluster"] = df["post_clust"].astype("category")
    df = df.loc[:,~df.columns.isin(['pre_clust','post_clust'])]
    df["same_cluster"] = df["same_clust"].astype("category") 
    
    # Deleting unwanted features
    df = df.loc[:, ~df.columns.isin(['ID',
                                    'same_clust',
                                    'comp',
                                    'poba',
                                    'prba',
                                    'projection_g',
                                    'pre_feature_weights',
                                    'post_feature_weights',
                                    'pre_morph_embeddings',
                                    'post_morph_embeddings',
                                    'pre_nucleus_id',
                                    'post_nucleus_id',
                                    ])]
    
    return df