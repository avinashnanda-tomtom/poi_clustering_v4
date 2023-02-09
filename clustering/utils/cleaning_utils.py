from w3lib.url import url_query_cleaner
from url_normalize import url_normalize
import re
from unidecode import unidecode
import numpy as np
from itertools import compress
import os
import shutil
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

patterns = 'home.html|index.html|home.shtml|index.shtml|.html|.shtml|\.com|\.ac|\.ad|\.ae|\.af|\.ag|\.ai|\.al|\.am|\.an|\.ao|\.aq|\.ar|\.as|\.asia|\.at|\.au|\.aw|\.ax|\.az|\.ba|\.bb|\.bd|\.be|\.bf|\.bg|\.bh|\.bi|\.bj|\.bl|\.bm|\.bn|\.bo|\.bq|\.br|\.bs|\.bt|\.bv|\.bw|\.by|\.bz|\.ca|\.cc|\.cd|\.cf|\.cg|\.ch|\.ci|\.ck|\.cl|\.cm|\.cn|\.co|\.cr|\.cu|\.cv|\.cw|\.cx|\.cy|\.cz|\.de|\.dj|\.dk|\.dm|\.do|\.dz|\.ec|\.edu|\.ee|\.eg|\.eh|\.er|\.es|\.et|\.eu|\.fi|\.fj|\.fk|\.fm|\.fo|\.fr|\.ga|\.gb|\.gd|\.ge|\.gf|\.gg|\.gh|\.gi|\.gl|\.gm|\.gn|\.gov|\.gp|\.gq|\.gr|\.gs|\.gt|\.gu|\.gw|\.gy|\.hk|\.hm|\.hn|\.hr|\.ht|\.hu|\.id|\.ie|\.il|\.im|\.in|\.io|\.iq|\.ir|\.is|\.it|\.je|\.jm|\.jo|\.jobs|\.jp|\.ke|\.kg|\.kh|\.ki|\.km|\.kn|\.kp|\.kr|\.kw|\.ky|\.kz|\.la|\.lb|\.lc|\.li|\.lk|\.lr|\.ls|\.lt|\.lu|\.lv|\.ly|\.ma|\.mc|\.md|\.me|\.mf|\.mg|\.mh|\.mk|\.ml|\.mm|\.mn|\.mo|\.mp|\.mq|\.mr|\.ms|\.mt|\.mu|\.mv|\.mw|\.mx|\.my|\.mz|\.na|\.nc|\.ne|\.nf|\.ng|\.ni|\.nl|\.no|\.np|\.nr|\.nu|\.nz|\.om|\.org|\.pa|\.pe|\.pf|\.pg|\.ph|\.pk|\.pl|\.pm|\.pn|\.pr|\.ps|\.pt|\.pw|\.py|\.qa|\.re|\.ro|\.rs|\.ru|\.rw|\.sa|\.sb|\.sc|\.sd|\.se|\.sg|\.sh|\.si|\.sj|\.sk|\.sl|\.sm|\.sn|\.so|\.sr|\.ss|\.st|\.su|\.sv|\.sx|\.sy|\.sz|\.tc|\.td|\.tf|\.tg|\.th|\.tj|\.tk|\.tl|\.tm|\.tn|\.to|\.tp|\.tr|\.tt|\.tv|\.tw|\.tz|\.ua|\.ug|\.uk|\.um|\.us|\.uy|\.uz|\.va|\.vc|\.ve|\.vg|\.vi|\.vn|\.vu|\.wf|\.ws|\.ye|\.yt|\.za|\.zm|\.zw'


def canonical_url(u):
    try:
        if u == u:
            u = url_normalize(u)
            u = url_query_cleaner(u,
                                parameterlist=[
                                    'utm_source', 'utm_medium', 'utm_campaign',
                                    'utm_term', 'utm_content'
                                ],
                                remove=True)

            if u.startswith("http://"):
                u = u[7:]
            if u.startswith("https://"):
                u = u[8:]
            if u.startswith("www."):
                u = u[4:]
            if u.endswith("/"):
                u = u[:-1]

            u = re.sub(patterns, r"", u)
            # u = re.sub(patterns, r"\2", u)

            return u
        else:
            return u
    except:
        return u


def combine_category(rawCat, insertedCat, preemptiveCat):
    """combine 3 categories two one

    Args:
        rawCat (list): raw categories.
        insertedCat (list): inserted categories.
        preemptiveCat (list): pre-emptive categories.

    Returns:
        list: concatenated category list
    """

    category = list(set(rawCat + preemptiveCat + insertedCat))

    category = [x for x in category if x]

    return category

def clean_alt_list(list_):
    """This is a function to format the list present in a pandas dataframe so that it can be exploded.

    Args:
        list_ (list): list cell present in a dataframe.

    Returns:
        dataframe: formats the list so that it can be exploded.
    """
    list_ = list_.replace("\"", "")
    list_ = list_.replace("\\", "")
    list_ = list_.replace("[[", "[")
    list_ = list_.replace("]]", "]")
    list_ = list_.replace("], [", ",")
    list_ = list_.replace("],", "")
    list_ = list_.replace(",[", "")
    list_ = list_.replace(", ", '","')
    list_ = list_.replace("[", '["')
    list_ = list_.replace("]", '"]')

    return list_



def clean_email(x):  # remove common words without much meaning
    if x == x:
        words = [
            'info', 'gmail', 'admin', 'customercare', 'office', 'reservations',
            'customerservices', 'enquiries', 'marketing', 'visit', 'bookings',
            'tourism', 'events', 'yahoo', 'sales', 'bookings',
            'customerservice', 'sales', 'customerrvices', 'webmail', 'hello',
            'contact', 'hotmail', 'support', 'stay', 'tourism', 'mweb'
        ]
        for word in words:
            x = x.replace(" " + word + " ", " ")  # middle
            if x[:len(word) + 1] == word + " ":  # start
                x = x[len(word) + 1:]
            if x[-len(word) - 1:] == " " + word:  # end
                x = x[:-len(word) - 1]
        x = re.sub("\d+", " ", x)
        return x
    else:
        return x


def extract_digits(text):
    L = []
    for char in text:
        if char.isdigit():
            L.append(char)
    res = "".join(L)[-10:].zfill(10)
    if len(res) > 0:
        return res
    else:
        return text


def process_phone(text):
    if text != '[]':
        text = str(text)
        text = text.split(",")
        text_l = []
        for tx in text:
            text_l.append(extract_digits(tx))
        return text_l
    else:
        return text


def unique_list(text_str):
    if text_str == text_str:
        l = text_str.split()
        temp = []
        for x in l:
            if x not in temp:
                temp.append(x)

        return ' '.join(temp)
    else:
        return text_str


def clean_streets(x):
    """expands few words to a uniform form present in streets and cities.

    Args:
        x (text): text to clean.

    Returns:
        text: text
    """
    if x == x:
        wwords = [
            ["corner", "cnr"],
            ["street", "str"],
            ["road", "rd"],
            ["avenue", "av", "ave"],
            ["highway", "hwy"],
            ["floor", "fl", "flr"],
            ["boulevard", "blvd", "blv"],
            ["center", "centre"],
            ["drive", "dr"],
            ["ste", "suite"],
            ["square", "sq"],
        ]
        for words in wwords:
            for word in words[1:]:
                x = x.replace(" " + word + " ", " " + words[0] + " ")  # middle
                if x[:len(word) + 1] == word + " ":  # start
                    x = x.replace(word + " ", words[0] + " ")
                if x[-len(word) - 1:] == " " + word:  # end
                    x = x.replace(" " + word, " " + words[0])
        return x
    else:
        return x


def convert_to_List_multilingual(df):
    """convert a list present as string in a dataframe to a list type so that it can be exploded.

    Args:
        df (dataframe): dataframe of raw poi.

    Returns:
        dataframe: dataframe with string list columns converted to list type.
    """
    df["latitude"] = df["latitude"].apply(eval)
    df["longitude"] = df["longitude"].apply(eval)
    str_cols = [
        "sourceNames",
        "rawCategories",
        "insertedCategories",
        "brands",
        "preemptiveCategories",
        "houseNumber",
        "streets",
        "cities",
        "postalCode",
        "script_names",
        "script_streets",
        "script_city"
    ]

    for col in str_cols:
        df[col] = df[col].str.lower()
        df[col] = df[col].parallel_apply(clean_alt_list)
        df[col] = df[col].apply(eval)

    return df

def latin_select(name_list,script_list):
    
    script_list = [x=="latn" for x in script_list]
    temp = list(compress(name_list, script_list))
    if len(temp)==0:
        return name_list
    else:
        return temp
    
def check_zero(alist):
    return all(ele == 0 for ele in alist)


def drop_zero_coord(data):
    data1 = data.copy()
    data1["latitude"]= data1["latitude"].apply(eval)
    data1["longitude"]= data1["longitude"].apply(eval)
    data["drop_lat"] = data1["latitude"].apply(check_zero)
    data["drop_long"] = data1["longitude"].apply(check_zero)
    data = data[~((data["drop_lat"]==True) | (data["drop_long"]==True))]
    data = data.drop(["drop_lat","drop_long"],axis=1)
    return data

def rem_words(x):  # remove common words without much meaning
    if x == x:
        words = [
            "the", "of", "an", "and", "at", "no", "by", "in", "co", "for",
            "inc", "llc", "llp", "ltd", "on", "to", "as", "is"
        ]
        for word in words:
            x = x.replace(" " + word + " ", " ")  # middle
            if x[:len(word) + 1] == word + " ":  # start
                x = x[len(word) + 1:]
            if x[-len(word) - 1:] == " " + word:  # end
                x = x[:-len(word) - 1]
        return x
    else:
        return x


def clean_name(x):
    """expand few words to a uniform form present in sourcenames

    Args:
        x (text): text

    Returns:
        _type_: uniform text
    """
    wwords = [
        ["street", "str"],
        ["road", "rd"],
        ["mount", "mt"],
        ["saint", "st"],
        ["floor", "fl", "flr"],
        ["boulevard", "blvd", "blv"],
        ["centre", "center"],
        ["doctor", "dr"],
        ["suite", "ste"],
        ["motor", "motors"],
        ["highway", "hwy"],
    ]
    for words in wwords:
        for word in words[1:]:
            x = x.replace(" " + word + " ", " " + words[0] + " ")  # middle
            if x[:len(word) + 1] == word + " ":  # start
                x = x.replace(word + " ", words[0] + " ")
            if x[-len(word) - 1:] == " " + word:  # end
                x = x.replace(" " + word, " " + words[0])
    return x


def clean_text(x):
    if x == x:
        x = unidecode(str(x))
        # lower case
        x = x.lower()
        x = re.sub("\.0$", "", x)
        x = x.replace("™", " ")
        x = x.replace("®", " ")
        x = x.replace("ⓘ", " ")
        x = x.replace("©", " ")
        # remove symbols
        x = x.replace('"', " ")
        ss = ",:;'/-+&()!#$%*.|\@`~^<>?[]{}_=\n"  # noqa
        for i in range(len(ss)):
            x = x.replace(ss[i], " ")

        x = re.sub(" +", " ", x)
        x = x.strip()
        return x
    else:
        return x
    
def convert_to_List(df):
    """convert a list present as string in a dataframe to a list type so that it can be exploded.

    Args:
        df (dataframe): dataframe of raw poi.

    Returns:
        dataframe: dataframe with string list columns converted to list type.
    """
    df["latitude"] = df["latitude"].apply(eval)
    df["longitude"] = df["longitude"].apply(eval)
    str_cols = [
        "sourceNames",
        "rawCategories",
        "insertedCategories",
        "brands",
        "preemptiveCategories",
        "houseNumber",
        "streets",
        "cities",
        "postalCode",
    ]

    for col in str_cols:
        df[col] = df[col].str.lower()
        df[col] = df[col].apply(clean_alt_list)
        df[col] = df[col].apply(eval)

    return df
    
def remove_path(path):
    """Remove file or folder.

    Args:
        path (text): path of folder or file to remove.

    Raises:
        ValueError: if path is not found.
    """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
