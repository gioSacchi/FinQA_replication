import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from time import time
import json
import re
import pandas as pd

stop_words = set(stopwords.words('english'))

def tokenize(text):
    # get all words in text
    words = re.findall(r'\b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b', text)
    return words

# Calculate the Word Mover's Distance between two sentences
def compare_sentences(sentence1, sentence2, model):
    # Tokenize the sentences
    sentence1_tokens = word_tokenize(sentence1.lower())
    sentence2_tokens = word_tokenize(sentence2.lower())

    # Remove the stop words
    sentence1_tokens = [token for token in sentence1_tokens if token not in stop_words]
    sentence2_tokens = [token for token in sentence2_tokens if token not in stop_words]

    # Calculate the Word Mover's Distance
    t1 = time()
    distance = model.wmdistance(sentence1_tokens, sentence2_tokens) 
    print("Time taken to calculate WMD: ", time() - t1)

    return distance

def main():
    # open result file
    res_file = r"C:\Users\pingu\FinQA_replication\res.txt" 
    with open(res_file) as infile:
        results_dict = json.load(infile)

    # load data
    data_file = r"C:\Users\pingu\FinQA_replication\dataset\test.json"
    df = pd.read_json(data_file)

    # load model
    word2vec_path = r'C:\Users\pingu\Desktop\GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    # score for each class
    class_scores = dict((res_class, 0) for res_class in results_dict)

    # iterate over df and make comparisons
    for index, row in df.iterrows():
        # print progress
        if index % 100 == 0:
            print("Progress: ", index, "/", len(df))

        # get question and gold inds
        question = row["qa"]["question"]
        golds = row["qa"]["gold_inds"]

        # make comparison and sum up distances
        sum = 0
        for text in golds.values():
            # make comparison
            sum += compare_sentences(question, text, model)
        # compute avg
        avg = sum / len(golds)

        # iterate over results and make comparisons
        for res_class, indecies in results_dict.items():
            # check if row id in indecies
            if row["id"] in indecies:
                # add average to class_scores
                class_scores[res_class] += avg
    
    # average class scores
    for res_class, score in class_scores.items():
        class_scores[res_class] = score / len(results_dict[res_class])

if __name__ == "__main__":
    main()
