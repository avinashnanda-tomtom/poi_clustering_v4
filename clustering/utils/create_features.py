import jellyfish
import nltk
import numpy as np
import pandas as pd
import rapidfuzz
import ray.util.multiprocessing as ray
from numpy.linalg import norm
from rapidfuzz import fuzz
from tqdm.auto import tqdm
from Config import config
from utils.cleaning_utils import clean_text


drop_cols = [
    "sourceNames1",
    "category1",
    "streets1",
    "cities1",
    "sourceNames2",
    "phoneNumbers1",
    "phoneNumbers2",
    "category2",
    "streets2",
    "cities2",
    "houseNumber1",
    "houseNumber2",
    "postalCode1",
    "postalCode2",
    "brands1",
    "brands2",
    "supplier1",
    "supplier2",
]


def manhattan(lat1, long1, lat2, long2):
    """Calculates manhattan distance

    Args:
        lat1 (float): latitude of point 1
        long1 (float): longitude of point 1
        lat2 (float): latitude of point 2
        long2 (float): longitude of point 2

    Returns:
        float: manhattan distance.
    """
    return np.abs(lat2 - lat1) + np.abs(long2 - long1)


def vectorized_haversine(lats1, lats2, longs1, longs2):
    """Calculates haversine distance vectorized.

    Args:
        lats1 (array): Array of latitude 1.
        lats2 (array): Array of longitude 1.
        longs1 (array): Array of latitude 2.
        longs2 (array): Array of longitude 2.

    Returns:
        array: array of haversine distance.
    """
    radius = 6371
    dlat = np.radians(lats2 - lats1)
    dlon = np.radians(longs2 - longs1)
    a = np.sin(
        dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lats1)) * np.cos(
            np.radians(lats2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c
    return d


def check_nan(a, b):
    if a == a and b == b:
        return True
    else:
        return False


def add_lat_lon_distance_features(df_pairs):
    """Calculates all lat long features for a dataframe with pairs of corrdinates.

    Args:
        df_pairs (dataframe): dataframe with pairs of poi from blocking.

    Returns:
        dataframe: dataframe with lat long features.
    """
    lat1 = df_pairs["latitude1"]
    lat2 = df_pairs["latitude2"]
    lon1 = df_pairs["longitude1"]
    lon2 = df_pairs["longitude2"]
    df_pairs["haversine"] = vectorized_haversine(lat1, lat2, lon1, lon2)
    col_64 = list(df_pairs.dtypes[df_pairs.dtypes == np.float64].index)
    for col in col_64:
        df_pairs[col] = df_pairs[col].astype(np.float32)
    return df_pairs


def fast_cosine(vec1, vec2, batch=10000, is_batch=True):
    """Function to calculate cosine similarity faster in vectorized form.

    Args:
        vec1 (array): array of embeddings1.
        vec2 (array): array of embeddings2.
        batch (int, optional): batch size of vectors to calculate cosine similarity. Defaults to 1000000.
        is_batch (bool, optional): To batch or not. Defaults to True.

    Returns:
        array: array of cosine similarity.
    """

    if is_batch:

        sims = np.empty((0), np.float32)

        for i in range(0, len(vec1), batch):
            cosine = np.sum(vec1[i:i + batch] * vec2[i:i + batch], axis=1)
            cosine = np.round(cosine, 3)
            sims = np.concatenate((sims, cosine))
        return sims
    else:
        cosine = np.sum(vec1 * vec2, axis=1)
        cosine = np.round(cosine, 3)
    return cosine


def strike_a_match(str1, str2):
    """Dice bigram calculation.

    Args:
        str1 (text): text1
        str2 (text): text2

    Returns:
        float: dice bigram score
    """
    if check_nan(str1, str2):
        pairs1 = set(nltk.bigrams(str1))
        pairs2 = set(nltk.bigrams(str2))
        union = len(pairs1) + len(pairs2)
        hit_count = len(pairs1.intersection(pairs2))
        try:
            return (2.0 * hit_count) / union
        except:
            if str1 == str2:
                return 1.0
            else:
                return 0.0
    else:
        return -1


def sorted_winkler(str1, str2):
    """find edit jaro wrinkler distance after sorting both the strings.

    Args:
        str1 (text): text1
        str2 (text): text2

    Returns:
        float: jarowinkler similarity of sorted strings.
    """
    if check_nan(str1, str2):
        a = sorted(str1.split(" "))
        b = sorted(str2.split(" "))
        a = " ".join(a)
        b = " ".join(b)
        return rapidfuzz.distance.JaroWinkler.similarity(a, b)
    else:
        return -1


def davies(str1, str2):
    """https://www.tandfonline.com/doi/full/10.1080/17538947.2017.1371253

    Args:
        str1 (text): text1
        str2 (text): text2

    Returns:
        float: score
    """
    if check_nan(str1, str2):
        a = str1.lower().replace("-", " ").split(" ")
        b = str2.lower().replace("-", " ").split(" ")
        for i in range(len(a)):
            if len(a[i]) > 1 or not (a[i].endswith(".")):
                continue
            replacement = len(str2)
            for j in range(len(b)):
                if b[j].startswith(a[i].replace(".", "")):
                    if len(b[j]) < replacement:
                        a[i] = b[j]
                        replacement = len(b[j])
        for i in range(len(b)):
            if len(b[i]) > 1 or not (b[i].endswith(".")):
                continue
            replacement = len(str1)
            for j in range(len(a)):
                if a[j].startswith(b[i].replace(".", "")):
                    if len(a[j]) < replacement:
                        b[i] = a[j]
                        replacement = len(a[j])
        a = set(a)
        b = set(b)
        aux1 = sorted_winkler(str1, str2)
        intersection_length = (sum(
            max(rapidfuzz.distance.JaroWinkler.similarity(i, j) for j in b)
            for i in a) + sum(
                max(
                    rapidfuzz.distance.JaroWinkler.similarity(i, j) for j in a)
                for i in b)) / 2.0
        aux2 = float(intersection_length) / (len(a) + len(b) -
                                             intersection_length)
        return (aux1 + aux2) / 2.0
    else:
        return -1


def get_phonetic_soundex(word1, word2):
    """Soundex phonetic similarity

    Args:
        word1 (text): word1
        word2 (text): word2

    Returns:
        float: soundex encoding jaro similarity
    """
    if check_nan(word1, word2):
        soundex_score = jellyfish.jaro_similarity(
            str(jellyfish.soundex(word1)), str(jellyfish.soundex(word2)))

        return soundex_score
    else:
        return -1


def get_phonetic_metaphone(word1, word2):
    """metaphone phonetic similarity

    Args:
        word1 (text): word1
        word2 (text): word2

    Returns:
        float: metaphone encoding jaro similarity
    """
    if check_nan(word1, word2):
        metaphone_score = jellyfish.jaro_similarity(
            str(jellyfish.metaphone(word1)), str(jellyfish.metaphone(word2)))

        return metaphone_score
    else:
        return -1


def lcs(text1, text2):
    if check_nan(text1, text2):
        return rapidfuzz.distance.LCSseq.similarity(str(text1), str(text2))/min(len(text1),len(text2))
    else:
        return -1


def jaro(text1, text2):
    if check_nan(text1, text2):
        return rapidfuzz.distance.Jaro.similarity(text1, text2)
    else:
        return -1


def leven(text1, text2):
    if check_nan(text1, text2):
        return rapidfuzz.distance.DamerauLevenshtein.normalized_similarity(
            text1, text2)
    else:
        return -1


def token_set_ratio(text1, text2):
    if check_nan(text1, text2):
        return fuzz.token_set_ratio(text1, text2) / 100
    else:
        return -1


def WRatio(text1, text2):
    if check_nan(text1, text2):
        return fuzz.WRatio(text1, text2) / 100
    else:
        return -1


def ratio(text1, text2):
    if check_nan(text1, text2):
        return fuzz.ratio(text1, text2) / 100
    else:
        return -1


def QRatio(text1, text2):
    if check_nan(text1, text2):
        return fuzz.QRatio(text1, text2) / 100
    else:
        return -1
