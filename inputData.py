import pyprind
import pandas as pd
import os
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords

def loadDataset(basePath = 'data'):
    labels = {'pos': 1, 'neg': 0}
    pbar = pyprind.ProgBar(50000)

    df = pd.DataFrame()

    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(basePath, s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()
                    df = df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()

    df.columns = ['review', 'sentiment']
    return df

def shuffleAndSave(dataSet):
    np.random.seed(0)
    df = dataSet.reindex(np.random.permutation(dataSet.index))
    df.to_csv('data/movic_data.csv', index=False, encoding='utf-8')


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) #[^]可以匹配任何不是^后边字符的其他字符。所以这个正则就表示了匹配所有的html标签

    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', '', text.lower()) + ' '.join(emoticons).replace('-', '')

    return text

def tokenizer(text):
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in text.split()]
    nltk.download('stopwords')
    stop = stopwords.words('english')

    return [w for w in tokens if w not in stop]


if __name__ == "__main__":
    # review_df = pd.read_csv('data/movie_data.csv', index_col=False)
    # review_df['review'] = review_df['review'].apply(preprocessor)
    print(tokenizer('a runner likes running and runs a lot'))