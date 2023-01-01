import nltk
# nltk.download('omw-1.4')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from time import time
import json
import re
import pandas as pd
import math
import statistics
import scipy.stats as stats

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def convert_to_wn_pos(pos):
    if pos.startswith("J"):
        return wn.ADJ
    elif pos.startswith("V"):
        return wn.VERB
    elif pos.startswith("N"):
        return wn.NOUN
    elif pos.startswith("R"):
        return wn.ADV
    elif pos == None:
        return None
    else:
        return ""

def tokenize(text):
    # get all words in text
    words = re.findall(r'\b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b', text)
    return words

def sentence_to_tokens(sentence):
    # Tokenize the sentence
    sentence_tokens = tokenize(sentence.lower())

    # Remove the stop words
    sentence_tokens = [token for token in sentence_tokens if token not in stop_words]

    return sentence_tokens

def sentence_to_tokens_and_lemmas(sentence):
    # Tokenize the sentence
    sentence_tokens = tokenize(sentence.lower())

    #words + tags for lemmatisation
    lemma_tags = pos_tag(sentence_tokens)

    # lemmatize sentence
    sentence_lemmas = []
    for word, tag in lemma_tags:
        tag = convert_to_wn_pos(tag)
        if tag != None and tag != "":
            sentence_lemmas.append(lemmatizer.lemmatize(word, tag))
        else:
            sentence_lemmas.append(lemmatizer.lemmatize(word))

    # idenfy indecies of stop words in sentence_tokens
    stop_word_indices = [i for i, x in enumerate(sentence_tokens) if x in stop_words]

    # Remove indecies of stop words from sentence_tokens and sentence_lemmas
    sentence_tokens = [token for i, token in enumerate(sentence_tokens) if i not in stop_word_indices]
    sentence_lemmas = [lemma for i, lemma in enumerate(sentence_lemmas) if i not in stop_word_indices]

    return sentence_tokens, sentence_lemmas

"""Calculate the Word Mover's Distance between two sentences"""
def WMD(sentence1, sentence2, model):
    # Tokenize the sentences
    sentence1_tokens = sentence_to_tokens(sentence1)
    sentence2_tokens = sentence_to_tokens(sentence2)

    # Calculate the Word Mover's Distance
    # t1 = time()
    distance = model.wmdistance(sentence1_tokens, sentence2_tokens) 
    # print("Time taken to calculate WMD: ", time() - t1)
    # print("WMD: ", distance)

    return distance

def intersect_ratio(sentence1, sentence2):
    # Tokenize the sentences
    sentence1_tokens = sentence_to_tokens(sentence1)
    sentence2_tokens = sentence_to_tokens(sentence2)

    # Calculate the intersection ratio
    intersection = list(set(sentence1_tokens).intersection(set(sentence2_tokens)))
    # intersection_ratio is ratio between the number of unique words appearing in both sentences 
    # and the total number of unique words appearing in both sentences
    intersection_ratio = len(intersection) / len(set(sentence1_tokens + sentence2_tokens))
    return intersection_ratio

def lemma_intersect_ratio(sentence1, sentence2):
    # Tokenize the sentences
    sentence1_tokens, sentence1_lemmas = sentence_to_tokens_and_lemmas(sentence1)
    sentence2_tokens, sentence2_lemmas = sentence_to_tokens_and_lemmas(sentence2)

    # Calculate the intersection ratio
    lemma_intersection = list(set(sentence1_lemmas).intersection(set(sentence2_lemmas)))
    # intersection_ratio is ratio between the number of unique words appearing in both sentences 
    # and the total number of unique words appearing in both sentences
    lemma_intersection_ratio = len(lemma_intersection) / len(set(sentence1_lemmas + sentence2_lemmas))
    return lemma_intersection_ratio

def stat_sign(data1, data2, alpha=0.05):
    # calculate mean and variance
    mean1 = statistics.mean(data1)
    mean2 = statistics.mean(data2)
    var1 = statistics.variance(data1)
    var2 = statistics.variance(data2)
    len1 = len(data1)
    len2 = len(data2)

    # calculate test statistic and p value, two-sided t-test, performs Welch’s t-test when variances are unequal
    t_stat, p_value = stats.ttest_ind_from_stats(mean1, math.sqrt(var1), len1, mean2, math.sqrt(var2), len2, equal_var=False, alternative='two-sided')
    deg_freedom = (var1/len1 + var2/len2)**2 / ((var1/len1)**2/(len1-1) + (var2/len2)**2/(len2-1))

    # check if p-value is less than alpha
    if p_value < alpha:
        significance = True
    else:
        significance = False

    return significance, t_stat, p_value, deg_freedom

def analysis(df, results_dict, model = None):
    # data for each class
    class_data = dict((res_class, []) for res_class in results_dict)

    inf_count = 0

    # iterate over df and make comparisons
    for index, row in df.iterrows():
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

            # make comparison with WMD
            # dist = WMD(question, text, model)

            # make comparison with intersection ratio
            dist = intersect_ratio(question, text)

            # if inf add to inf_count 
            if dist == float("inf"):
                inf_count += 1
            else:
                # online average update
                count += 1
                avg = dist/count + avg*(count-1)/count

        # iterate over results and make comparisons
        for res_class, indecies in results_dict.items():
            # check if row id in indecies, if avg is not inf and if count is not 0 (i.e. updates to avg have been made)
            if row["id"] in indecies and avg != float("inf") and count != 0:
                # add to class_data
                class_data[res_class].append(avg)

    # average class data
    class_scores = dict((res_class, 0) for res_class in results_dict)
    for res_class, data in class_data.items():
        class_scores[res_class] = statistics.mean(data)

    # print results
    print("class_scores: ", class_scores)
    print("class_counts: ", {res_class: len(data) for res_class, data in class_data.items()})
    print("inf_count: ", inf_count)

    return class_data

"""uses intersection ratio to calculate the max distance between question and gold inds"""
def analysis_max_ind(df, results_dict, model = None):
    # data for each class
    class_data = dict((res_class, []) for res_class in results_dict)

    inf_count = 0

    # iterate over df and make comparisons
    for index, row in df.iterrows():
        # get question and gold inds
        question = row["qa"]["question"]
        golds = row["qa"]["gold_inds"]

        # make comparison and sum up distances
        scores = []
        for key, text in golds.items():
            # skip if table
            if "table" in key:
                continue

            # make comparison with intersection ratio
            dist = intersect_ratio(question, text)

            # if inf add to inf_count 
            if dist == float("inf"):
                inf_count += 1
            else:
                scores.append(dist)
        
        # get max score
        max_val = None
        if scores != []:
            max_val = max(scores)

        # iterate over results and make comparisons
        for res_class, indecies in results_dict.items():
            # check if row id in indecies, max_val has been updated
            if row["id"] in indecies and max_val != None:
                # add to class_data
                class_data[res_class].append(max_val)

    # average class data
    class_scores = dict((res_class, 0) for res_class in results_dict)
    for res_class, data in class_data.items():
        class_scores[res_class] = statistics.mean(data)

    # print results
    print("class_scores: ", class_scores)
    print("class_counts: ", {res_class: len(data) for res_class, data in class_data.items()})
    print("inf_count: ", inf_count)

    return class_data

def analysis_max_ind_lemma(df, results_dict, model = None):
    # data for each class
    class_data = dict((res_class, []) for res_class in results_dict)

    inf_count = 0

    # iterate over df and make comparisons
    for index, row in df.iterrows():
        # get question and gold inds
        question = row["qa"]["question"]
        golds = row["qa"]["gold_inds"]

        # make comparison and sum up distances
        scores = []
        for key, text in golds.items():
            # skip if table
            if "table" in key:
                continue

            # make comparison with intersection ratio
            dist = lemma_intersect_ratio(question, text)

            # if inf add to inf_count 
            if dist == float("inf"):
                inf_count += 1
            else:
                scores.append(dist)
        
        # get max score
        max_val = None
        if scores != []:
            max_val = max(scores)

        # iterate over results and make comparisons
        for res_class, indecies in results_dict.items():
            # check if row id in indecies, max_val has been updated
            if row["id"] in indecies and max_val != None:
                # add to class_data
                class_data[res_class].append(max_val)

    # average class data
    class_scores = dict((res_class, 0) for res_class in results_dict)
    for res_class, data in class_data.items():
        class_scores[res_class] = statistics.mean(data)

    # print results
    print("class_scores: ", class_scores)
    print("class_counts: ", {res_class: len(data) for res_class, data in class_data.items()})
    print("inf_count: ", inf_count)

    return class_data

"""uses WMD to calculate the min distance between question and gold inds"""
def analysis_min_ind(df, results_dict, model = None):
    # data for each class
    class_data = dict((res_class, []) for res_class in results_dict)

    inf_count = 0

    # iterate over df and make comparisons
    for index, row in df.iterrows():
        # get question and gold inds
        question = row["qa"]["question"]
        golds = row["qa"]["gold_inds"]

        # make comparison and sum up distances
        scores = []
        for key, text in golds.items():
            # skip if table
            if "table" in key:
                continue

            # make comparison with WMD
            dist = WMD(question, text, model)

            # if inf add to inf_count 
            if dist == float("inf"):
                inf_count += 1
            else:
                scores.append(dist)
        
        # get min score
        min_val = None
        if scores != []:
            min_val = min(scores)

        # iterate over results and make comparisons
        for res_class, indecies in results_dict.items():
            # check if row id in indecies, min_val has been updated
            if row["id"] in indecies and min_val != None:
                # add to class_data
                class_data[res_class].append(min_val)

    # average class data
    class_scores = dict((res_class, 0) for res_class in results_dict)
    for res_class, data in class_data.items():
        class_scores[res_class] = statistics.mean(data)

    # print results
    print("class_scores: ", class_scores)
    print("class_counts: ", {res_class: len(data) for res_class, data in class_data.items()})
    print("inf_count: ", inf_count)

    return class_data

def main():
    # open result file
    res_file = r"C:\Users\pingu\FinQA_replication\dataset_analysis\res.txt" 
    with open(res_file) as infile:
        results_dict = json.load(infile)

    # load data
    data_file = r"C:\Users\pingu\FinQA_replication\dataset\test.json"
    df = pd.read_json(data_file)

    # load model
    # t1 = time()
    # word2vec_path = r'C:\Users\pingu\Desktop\GoogleNews-vectors-negative300.bin.gz'
    # model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    # print("Time taken to load model: ", time() - t1)
    # make analysis
    # class_data = analysis(df, results_dict, model)
    # class_data = analysis_min_ind(df, results_dict, model)

    # make analysis
    # class_data = analysis(df, results_dict)
    class_data = analysis_max_ind_lemma(df, results_dict)
    

    # test significance of results (t-test) on correct and incorrect
    correct = class_data["correct"]
    incorrect = class_data["incorrect"]
    significance, t_stat, p_value, deg_freedom = stat_sign(correct, incorrect, alpha=0.05)
    print("significance: ", significance)
    print("t_stat: ", t_stat)
    print("deg_freedom: ", deg_freedom)
    print("p_value: ", p_value, "alpha: ", 0.05)

if __name__ == "__main__":
    main()
