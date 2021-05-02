import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# reads the file, extracts headers and texts
def read_file():
    with open('news.xml', 'r') as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        headers = soup.find_all('value', {'name': 'head'})
        texts = soup.find_all('value', {'name': 'text'})
        return headers, texts


HEADERS, TEXTS = read_file()
STOPWORDS = stopwords.words('english')
PUNCTUATION = list(string.punctuation)


# tokenizes all texts
def tokenize_texts():
    tokenized_texts = []
    for text in TEXTS:
        text_list = word_tokenize(text.text.lower())
        tokenized_texts.append(text_list)
    return tokenized_texts


# lemmatizes texts
def lemmatize_text(tokenized_text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_texts = []
    for text in tokenized_text:
        lemmatized_text = []
        for word in text:
            new_word = lemmatizer.lemmatize(word)
            if word_checker(new_word):
                lemmatized_text.append(new_word)
        lemmatized_texts.append(lemmatized_text)
    return lemmatized_texts


# checks if that's a valid word
def word_checker(word):
    tag = nltk.pos_tag([word])[0][1]
    if word not in STOPWORDS and word not in PUNCTUATION and tag == "NN":
        return True
    return False


# prepares a dataset for tf_idf
def make_dataset(lemmatized_texts):
    dataset = []
    for text in lemmatized_texts:
        new_text = ' '.join(text)
        dataset.append(new_text)
    return dataset


# makes tf_idf matrix using TfidfVectorizer
def make_tfidf_matrix():
    dataset = make_dataset(lemmatize_text(tokenize_texts()))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset)
    terms = vectorizer.get_feature_names()
    return tfidf_matrix, terms


# makes pandas dataframe to make searching easier
def make_df():
    matrix, terms = make_tfidf_matrix()
    tfidf_df = pd.DataFrame(matrix.toarray(), columns=terms)
    return tfidf_df


# finds most common words in the df
def find_most_common():
    df = make_df()
    most_common = []
    for n in range(0, 10):
        row = df.iloc[n]
        sorted_values = dict(sorted(row.items(), key=lambda x : (x[1], x[0]) , reverse=True))
        top = list(sorted_values.keys())
        most_common.append(top[:5])
    return most_common


# prints results in the prescribed way
def printer(most_common):
    for header, words in zip(HEADERS, most_common):
        print(header.text + ":")
        for word in words:
            print(word, end=" ")
        print('\n')


printer(find_most_common())
