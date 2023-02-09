from flashtext import KeywordProcessor
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm.auto import tqdm
from utils.cat_distance_map import cat_dist_map
from utils.related_cat_map import related_cat
import re
from utils.create_features import  WRatio, lcs, token_set_ratio
import itertools

keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list([
    "east", "west", "north", "south", "north east", "north west", "south east",
    "south west", "northeast", "northwest", "southeast", "southwest",
    "south africa", "northbound", "north bound", "southbound", "south bound",
    "westbound", "west bound", "eastbound", "east bound"
])


def extract_directions(text):
    directions = keyword_processor.extract_keywords(text)
    directions = [
        x for x in directions if x not in ["south africa", "south african"]
    ]
    directions = list(set(directions))
    return directions


def lcs_range(value):
    if value>=0.8:
        return 1
    elif 0.5 <= value <0.8:
        return 2
    elif 0 <= value <0.5:
        return 3
    else:
        return 4
    


def phone_category(ph_l1,ph_l2):
    if ph_l1 == ph_l1 and ph_l2 == ph_l2:
        if len(ph_l1)==1 and  len(ph_l2)==1:
            return lcs_range(lcs(ph_l1[0],ph_l2[0]))
        elif len(ph_l1)==0 or  len(ph_l2)==0:
            return 4
        else:
            ph_tuple = list(itertools.product(ph_l1,ph_l2))
            lcs_value = max(list((lcs(*itup) for itup in ph_tuple)))
            return lcs_range(lcs_value)
    else:
        return 4
        
def phone_lcs(ph_l1,ph_l2):
    if ph_l1 == ph_l1 and ph_l2 == ph_l2:
        if len(ph_l1)==1 and  len(ph_l2)==1:
            return lcs(ph_l1[0],ph_l2[0])
        elif len(ph_l1)==0 or  len(ph_l2)==0:
            return -1
        else:
            ph_tuple = list(itertools.product(ph_l1,ph_l2))
            
            return max(list((lcs(*itup) for itup in ph_tuple)))
    else:
        return -1

def is_direction_match(directions1, directions2):
    if len(directions1) == 0 and len(directions2) == 0:
        return 1
    if len(directions1) == 0 or len(directions2) == 0:
        return 2

    elif any(i in directions1 for i in directions2):
        return 1

    else:
        return 0


def name_distance(name1, name2, batch=300000):
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 64
    sims = np.empty((0), np.float32)
    for i in tqdm(range(0, len(name1), batch)):
        embeddings1 = model.encode(name1[i:i + batch],
                                   batch_size=64,
                                   show_progress_bar=True,
                                   normalize_embeddings=True)
        embeddings2 = model.encode(name2[i:i + batch],
                                   batch_size=64,
                                   normalize_embeddings=True)
        cosine = np.sum(embeddings1 * embeddings2, axis=1)
        cosine = np.round(cosine, 3)
        sims = np.concatenate((sims, cosine))
    return sims


def is_related_cat(cat1, cat2):
    if cat1 == cat2:
        return 1
    for grp in related_cat:
        if cat1 in grp and cat2 in grp:
            return 1

    return 0


def category_match(num1, num2):
    if num1 == num2:
        return 1

    elif num1 != num2:
        return 0


def map_dist(text):
    if text not in cat_dist_map:
        return 0.6
    else:
        return cat_dist_map[text]


def brand_match(text1, text2):
    if text1 != text1 and text2 != text2:
        return 3
    elif text1 != text1 and text2 == text2:
        return 2
    elif text1 == text1 and text2 != text2:
        return 2
    elif text1 == text2:
        return 1
    elif token_set_ratio(text1, text2) >= 0.8:
        return 1
    else:
        return 0


def house_match(text1, text2):
    if text1 != text1 or text2 != text2:
        return 2
    elif text1 == text2:
        return 1
    else:
        return 0


cat_subcat_map = [
    ['bar', 'cafe', 'pub'], ['cafe', 'coffee shop', 'tea house', 'cafeterias'],
    ['florists', 'house garden garden centers services'],
    ['college university', 'junior college community college'],
    ['convenience stores', 'food drinks grocers', 'food drinks wine spirits'],
    ['hardware', 'house garden do it yourself centers'],
    ['high school', 'primary school'],
    ['beauty salon', 'hairdressers barbers', 'beauty supplies'],
    ['personal care facility', 'personal service'],
    ['food drinks grocers', 'supermarkets hypermarkets'],
    ['bed breakfast guest houses', 'hotel', 'cottage'],
    ['dry cleaners', 'laundry'],
    ['flats apartment complex', 'townhouse complex'],
    ['fast food', 'pizza', 'seafood', 'sandwich', 'grill'],
    ['food drinks other food shops', 'specialty foods'],
    ['diversified financials', 'tax services'],
    ['child care facility', 'pre school'], ['business services', 'service'],
    ['cabins lodges', 'resort'], ['caravan site', 'rest camps']
]


def check_subcat_map(text1, text2):
    for cat_g in cat_subcat_map:
        if (text1 in cat_g) and (text2 in cat_g):
            return True


def sub_category_match(text1, text2):
    if text1 != text1 or text2 != text2:
        return 2
    elif text1 == text2:
        return 1
    elif token_set_ratio(text1, text2) >= 0.8:
        return 1
    elif check_subcat_map(text1, text2):
        return 1
    else:
        return 0


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


def email_url_match(text1, text2):
    if text1 != text1 or text2 != text2:
        return 2
    elif text1 == text2:
        return 1
    elif token_set_ratio(text1, text2) >= 0.8 or WRatio(text1, text2) >= 0.9:
        return 1
    else:
        return 0


def name_number_match(text1, text2):
    if text1 != text1 and text2 != text2:
        return 3
    elif text1 != text1 and text2 == text2:
        return 2
    elif text1 == text1 and text2 != text2:
        return 2
    elif text1 == text2:
        return 1
    else:
        return 0
    
    
def clean_email(x):  # remove common words without much meaning
    if x == x:
        words = [
            'info', 'gmail', 'admin', 'customercare', 'office', 'reservations',
            'customerservices', 'enquiries', 'marketing', 'visit', 'bookings',
            'tourism', 'events', 'yahoo', 'sales', 'bookings',
            'customerservice', 'sales', 'customerrvices', 'webmail', 'hello',
            'contact', 'hotmail', 'support', 'stay', 'tourism','mweb'
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