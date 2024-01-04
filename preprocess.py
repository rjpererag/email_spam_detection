import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def lowerColumnsNames(df:pd.DataFrame):
    original_colmuns = df.columns.values.tolist()
    new_columns = [column.lower().strip() for column in original_colmuns]
    df_processed = df.rename(columns=dict(zip(original_colmuns, new_columns)))

    return df_processed

def textProcessing(text:str) ->str:

    #removing urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text_no_url = url_pattern.sub('', text)

    #lowecasing
    text_lower = text_no_url.lower()

    #removing punctuation
    text_no_punct = "".join(char for char in text_lower if char not in string.punctuation)

    #removing stop words
    stop_words = set(stopwords.words('english'))
    text_no_stop = " ".join(char for char in text_no_punct.split(" ") if char not in stop_words)

    return text_no_stop

def textSeriesProcessing(series:pd.Series):
    return series.apply(textProcessing)


def labelProcessing(label_series:pd.Series) -> np.array :

    label_encoder = LabelEncoder()
    data_label_encoder = label_encoder.fit_transform(label_series)

    one_hot_encoder = OneHotEncoder(sparse = False)
    data_label_hot_encoder = one_hot_encoder.fit_transform(data_label_encoder.reshape(-1,1))

    return data_label_hot_encoder, data_label_encoder

