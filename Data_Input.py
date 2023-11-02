import pandas as pd
import numpy as np

def Input_Data(verbose=False):
    #load in training data on each potential synapse
    data = pd.read_csv("./train_data.csv")

    #load in additional features for each neuron
    feature_weights = pd.read_csv("./feature_weights.csv")
    morph_embeddings = pd.read_csv("./morph_embeddings.csv")
    
    if (verbose):
        print("Main Dataframe Size:",data.shape)
        print("\nMain Dataframe:")
        data.info()
        print("Main Dataframe Size:",data.shape)
        print("\nMain Dataframe:")
        data.info()
        print("\nFeature Weights Size:",feature_weights.shape)
        print("\nFeature Weights:")
        feature_weights.info(verbose=False)
        print("\nMorphological Embeddings Size:",morph_embeddings.shape)
        print("\nMorphological Embeddings:")
        morph_embeddings.info(verbose=False)

    # join all feature_weight_i columns into a single np.array column
    feature_weights["feature_weights"] = (
            feature_weights.filter(regex="feature_weight_")
            .sort_index(axis=1)
            .apply(lambda x: np.array(x), axis=1)
            )
    # delete the feature_weight_i columns
    feature_weights.drop(
            feature_weights.filter(regex="feature_weight_").columns, axis=1, inplace=True
            )
    
    # join all morph_embed_i columns into a single np.array column
    morph_embeddings["morph_embeddings"] = (
            morph_embeddings.filter(regex="morph_emb_")
            .sort_index(axis=1)
            .apply(lambda x: np.array(x), axis=1)
            )
    # delete the morph_embed_i columns
    morph_embeddings.drop(
            morph_embeddings.filter(regex="morph_emb_").columns, axis=1, inplace=True
            )
    data = (
            data.merge(
                feature_weights.rename(columns=lambda x: "pre_" + x), 
                how="left", 
                validate="m:1",
                copy=False,
                )
            .merge(
                feature_weights.rename(columns=lambda x: "post_" + x),
                how="left",
                validate="m:1",
                copy=False,
                )
            .merge(
                morph_embeddings.rename(columns=lambda x: "pre_" + x),
                how="left",
                validate="m:1",
                copy=False,
                )
            .merge(
                morph_embeddings.rename(columns=lambda x: "post_" + x),
                how="left",
                validate="m:1",
                copy=False,
                )
            )
    if (verbose):
        print("Data Size:",data.shape)
        print("\nData:")
        data.info()

    return data
