import pandas as pd
import numpy as np
import math

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
