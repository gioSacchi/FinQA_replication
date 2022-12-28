import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
# from gensim.models import Word2Vec
# from gensim.test.utils import common_texts
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim import models
from time import time

word2vec_path = r'C:\Users\pingu\Desktop\GoogleNews-vectors-negative300.bin.gz'
w2v_model = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
model = w2v_model
stop_words = set(stopwords.words('english'))

# Calculate the Word Mover's Distance between two sentences
def compare_sentences(sentence1, sentence2):
    # Tokenize the sentences
    sentence1_tokens = word_tokenize(sentence1.lower())
    sentence2_tokens = word_tokenize(sentence2.lower())

    # Remove the stop words
    sentence1_tokens = [token for token in sentence1_tokens if token not in stop_words]
    sentence2_tokens = [token for token in sentence2_tokens if token not in stop_words]
    
    # Load the model
    # model = Word2Vec.load("word2vec.model")
    # model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)

    # Calculate the Word Mover's Distance
    t1 = time()
    distance = model.wmdistance(sentence1_tokens, sentence2_tokens) 
    print("Time taken to calculate WMD: ", time() - t1)
    print("Distance between sentences: ", distance) 

    return distance

question = "for acquired customer-related and network location intangibles , what is the expected annual amortization expenses , in millions?"
gold1 = "american tower corporation and subsidiaries notes to consolidated financial statements ( 3 ) consists of customer-related intangibles of approximately $ 75.0 million and network location intangibles of approximately $ 72.7 million ."
gold2 =  "the customer-related intangibles and network location intangibles are being amortized on a straight-line basis over periods of up to 20 years ."

d1 = compare_sentences(question, gold1)
d2 = compare_sentences(question, gold2)
print("Avg ", (d1 + d2) / 2)