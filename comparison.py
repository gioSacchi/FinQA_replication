import nltk
# nltk.download('omw-1.4')
# nltk.download('stopwords')
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
    sentence1_tokens = tokenize(sentence1.lower())
    sentence2_tokens = tokenize(sentence2.lower())

    # Remove the stop words
    sentence1_tokens = [token for token in sentence1_tokens if token not in stop_words]
    sentence2_tokens = [token for token in sentence2_tokens if token not in stop_words]

    # Calculate the Word Mover's Distance
    # t1 = time()
    distance = model.wmdistance(sentence1_tokens, sentence2_tokens) 
    # print("Time taken to calculate WMD: ", time() - t1)
    # print("WMD: ", distance)

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
    t1 = time()
    word2vec_path = r'C:\Users\pingu\Desktop\GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    print("Time taken to load model: ", time() - t1)

    # score for each class
    class_scores = dict((res_class, 0) for res_class in results_dict)

    inf_count = 0

    class_counts = dict((res_class, 0) for res_class in results_dict)

    # iterate over df and make comparisons
    for index, row in df.iterrows():
        # print progress
        if index % 100 == 0:
            print("Progress: ", index, "/", len(df))

        # get question and gold inds
        question = row["qa"]["question"]
        golds = row["qa"]["gold_inds"]

        # make comparison and sum up distances
        avg = 0
        count = 0
        for key, text in golds.items():
            # skip if table
            if "table" in key:
                continue

            # make comparison
            dist = compare_sentences(question, text, model)
            # if inf add to inf_count 
            if dist == float("inf"):
                inf_count += 1
            else:
                # online average update
                count += 1
                avg = dist/count + avg*(count-1)/count

        # iterate over results and make comparisons
        for res_class, indecies in results_dict.items():
            # check if row id in indecies
            if row["id"] in indecies and avg != float("inf") and count != 0:
                # add average to class_scores
                class_scores[res_class] += avg
                class_counts[res_class] += 1
    
    # average class scores
    for res_class, score in class_scores.items():
        class_scores[res_class] = score / class_counts[res_class]
    
    # print results
    print("class_scores: ", class_scores)
    print("class_counts: ", class_counts)
    print("inf_count: ", inf_count)

if __name__ == "__main__":
    main()
