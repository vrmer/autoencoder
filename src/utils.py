import random
import glob
import tqdm
import pickle5 as pickle
# import pickle
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def compare_embeddings(original_embeddings, reconstructed_embeddings):
    """
    Compares the original embeddings with the versions that were reconstructed by the Autoencoder.
    Returns the average cosine similarity score.

    :param original_embedding: Tensors in float16 format
    :param reconstructed_embedding: Tensors in float16 format
    :return: average cosine similarity in float
    """
    similarities = []

    with tqdm.tqdm(total=len(original_embeddings), desc='Comparing embeddings: ') as pbar:
        for original_embedding, reconstructed_embedding in zip(original_embeddings, reconstructed_embeddings):
            cs = cosine_similarity([original_embedding], [reconstructed_embedding])[0][0]

            similarities.append(cs)

            pbar.update(1)

    return similarities


def load_lists(config_line, symbol_to_split=','):
    """
    Loads lists from a config file.

    :param config_line: line in the config file
    :param symbol_to_split: by default it is a comma
    :return: a list
    """
    string_list = config_line.split(symbol_to_split)
    try:
        loaded_list = [int(item) for item in string_list]
    except ValueError:
        try:
            loaded_list = [float(item) for item in string_list]
        except ValueError:
            loaded_list = [str(item).strip() for item in string_list]
    return loaded_list


def load_data(glob_pattern, embeddings_type, sample=None, top=False, percentage=False):
    """
    Read data points (embeddings) from FBK splits.

    :param glob_pattern: pattern for input files
    :param sample: if True, return a random subset of the data sets
    :param top: if True, take sample from the top of each training set
    :param percentage: if True, evenly sample the given percentage of each training set
    :return: data points and labels.
    """

    sentences = list()
    embeddings = list()
    labels = list()

    for filename in glob.glob(glob_pattern):
        # df = pd.read_pickle(filename)
        with open(filename, "rb") as infile:
            df = pickle.load(infile)

        # extract data
        sents = list(df["Sentence"])
        embs = list(df[f"{embeddings_type}"])
        lbs = list(df["TAUS_category"])

        del df

        if sample:
            if percentage is True:
                # fix random seed so that the same items are picked every time
                random.seed(1995)
                count = int(len(sents) / sample)
                print(count)
                selection = random.sample(list(zip(sents, embs, lbs)), count)
            elif len(sents) < sample:
                selection = list(zip(sents, embs, lbs))
            else:
                if top:
                    selection = list(zip(sents, embs, lbs))[:sample]
                else:
                    selection = random.sample(list(zip(sents, embs, lbs)), sample)
            sents, embs, lbs = zip(*selection)

        sentences.append(sents)
        embeddings.append(embs)
        labels.append(lbs)

    # print(df.info())

    # flatten lists
    sentences = [sent for sublist in sentences for sent in sublist]
    embeddings = [vector for sublist in embeddings for vector in sublist]
    labels = [label for sublist in labels for label in sublist]

    return sentences, tf.convert_to_tensor(embeddings), np.array(labels)


def load_data_outer(glob_pattern, embeddings_type, no_of_samples):

    if no_of_samples.endswith('%'):
        sample = int(no_of_samples.rstrip('%'))
        sents, x, y = load_data(glob_pattern, embeddings_type, sample=int(sample), top=False, percentage=True)
    else:
        try:
            sents, x, y = load_data(glob_pattern, embeddings_type, sample=int(no_of_samples), top=True)
        except:
            sents, x, y = load_data(glob_pattern, embeddings_type, sample=None, top=True)

    return sents, x, y


def plot_history(history, metric, experiment_name, plots_dir):
    """ Plot model performance across training epochs. """

    plt.plot(history.history[metric])
    plt.plot(history.history[f"val_{metric}"])
    plt.title(f"model {metric}")
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    os.makedirs(f"{plots_dir}/{experiment_name}", exist_ok=True)
    plt.savefig(f"{plots_dir}/{experiment_name}/{experiment_name}_{metric}.png")
    plt.close()

    pass
