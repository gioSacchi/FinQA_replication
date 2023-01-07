from gensim.models import KeyedVectors
from time import time
import json
import pandas as pd
import math
import statistics
import scipy.stats as stats
from anal_util import *

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

def jaccard_similarity(sentence1, sentence2):
    # Tokenize the sentences
    sentence1_tokens = sentence_to_tokens(sentence1)
    sentence2_tokens = sentence_to_tokens(sentence2)

    # Calculate the intersection ratio
    intersection = list(set(sentence1_tokens).intersection(set(sentence2_tokens)))
    # intersection_ratio is ratio between the number of unique words appearing in both sentences 
    # and the total number of unique words appearing in both sentences
    intersection_ratio = len(intersection) / len(set(sentence1_tokens + sentence2_tokens))
    return intersection_ratio

def lemma_jaccard_similarity(sentence1, sentence2):
    # Tokenize the sentences
    sentence1_tokens, sentence1_lemmas = sentence_to_tokens_and_lemmas(sentence1)
    sentence2_tokens, sentence2_lemmas = sentence_to_tokens_and_lemmas(sentence2)

    # Calculate the intersection ratio
    lemma_intersection = list(set(sentence1_lemmas).intersection(set(sentence2_lemmas)))
    lemma_union = list(set(sentence1_lemmas).union(set(sentence2_lemmas)))
    # intersection_ratio is ratio between the number of unique words appearing in both sentences 
    # and the total number of unique words appearing in both sentences
    lemma_jacc_sim = len(lemma_intersection) / len(lemma_union)
    return lemma_jacc_sim

def lemma_jaccard_distance(sentence1, sentence2):
    # Tokenize the sentences
    sentence1_tokens, sentence1_lemmas = sentence_to_tokens_and_lemmas(sentence1)
    sentence2_tokens, sentence2_lemmas = sentence_to_tokens_and_lemmas(sentence2)

    # Calculate the intersection ratio
    lemma_intersection = list(set(sentence1_lemmas).intersection(set(sentence2_lemmas)))
    lemma_union = list(set(sentence1_lemmas).union(set(sentence2_lemmas)))
    # intersection_ratio is ratio between the number of unique words appearing in both sentences 
    # and the total number of unique words appearing in both sentences
    lemma_jacc_dist = (len(lemma_union) - len(lemma_intersection)) / len(lemma_union)
    return lemma_jacc_dist

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
            dist = jaccard_similarity(question, text)

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
            dist = jaccard_similarity(question, text)

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
            dist = lemma_jaccard_similarity(question, text)

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

def total_max_ind_lemma(df):
    data_vals = []
    inf_count = 0

    # iterate over df and make comparisons
    for index, row in df.iterrows():
        # get question and gold inds
        question = row["qa"]["question"]
        golds = row["qa"]["gold_inds"]

        # make comparison and sum up distances
        scores = []
        for key, text in golds.items():
            # # skip if table
            # if "table" in key:
            #     continue

            # make comparison with intersection ratio
            dist = lemma_jaccard_similarity(question, text)

            # if inf add to inf_count 
            if dist == float("inf"):
                inf_count += 1
            else:
                scores.append(dist)
        
        # get max score
        max_val = None
        if scores != []:
            max_val = max(scores)

        if max_val != None:
            # add to class_data
            data_vals.append(max_val)

    # average data
    score = statistics.mean(data_vals)

    # print results
    print("score: ", score)
    print("count: ", len(data_vals))
    print("inf_count: ", inf_count)

    return score, data_vals

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

def nr_gold_inds_analysis(df, results_dict):
    # data for each class
    class_data = dict((res_class, []) for res_class in results_dict)

    # iterate over df and make comparisons
    for index, row in df.iterrows():
        # get question and gold inds
        nr = len(row["qa"]["gold_inds"])

        # iterate over results and make comparisons
        for res_class, indecies in results_dict.items():
            # check if row id in indecies, min_val has been updated
            if row["id"] in indecies:
                # add to class_data
                class_data[res_class].append(nr)

    # average class data
    class_scores = dict((res_class, 0) for res_class in results_dict)
    for res_class, data in class_data.items():
        class_scores[res_class] = statistics.mean(data)

    # print results
    print("class_scores: ", class_scores)
    print("class_counts: ", {res_class: len(data) for res_class, data in class_data.items()})

    return class_data

def input_type_anal(df, results_dict):
    # log how many gold_inds for various caterories
    log = {"text": [], "text-table": [], "table": []}

    # iterate over df and make comparisons
    for index, row in df.iterrows():
        # get gold inds
        golds = row["qa"]["gold_inds"]
        text = False
        table = False
        # figure out which gold inds types
        for gold_key in golds.keys():
            if "text" in gold_key:
                text = True
            elif "table" in gold_key:
                table = True
            else:
                print("Strange key", gold_key, index)
        
        # append nr gold inds
        if text:
            if table:
                in_type = "text-table"
            else:
                in_type = "text"
        else:
            in_type = "table"

        if row["id"] in results_dict["correct"]:
            log[in_type].append(1)
        elif row["id"] in results_dict["incorrect"]:
            log[in_type].append(0)
        else:
            continue

    # average class data
    class_scores = dict((res_class, 0) for res_class in log)
    for res_class, data in log.items():
        class_scores[res_class] = statistics.mean(data)

    # print results
    print("class_scores: ", class_scores)
    print("class_counts: ", {res_class: len(data) for res_class, data in log.items()})

    return log

def training_nr_gold_ind(trainset_path):
    df = pd.read_json(trainset_path)

    # log how many gold_inds for various caterories
    log = {"text": [], "text-table": [], "table": []}

    # iterate over data
    for index, row in df.iterrows():
        # get gold inds
        golds = row["qa"]["gold_inds"]
        text = False
        table = False
        # figure out which gold inds types
        for gold_key in golds.keys():
            if "text" in gold_key:
                text = True
            elif "table" in gold_key:
                table = True
            else:
                print("Strange key", gold_key, index)
        
        # append nr gold inds
        if text:
            if table:
                log["text-table"].append(len(golds))
            else:
                log["text"].append(len(golds))
        else:
            log["table"].append(len(golds))
    
    # make calculations 
    class_scores = dict((res_class, 0) for res_class in log)
    for res_class, data in log.items():
        class_scores[res_class] = statistics.mean(data)

    # print results
    print("class_scores: ", class_scores)
    print("class_counts: ", {res_class: len(data) for res_class, data in log.items()})            

def WDM_analysis(df, results_dict):
    # load model
    t1 = time()
    word2vec_path = r'C:\Users\pingu\Desktop\GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    print("Time taken to load model: ", time() - t1)
    # make analysis
    # class_data = analysis(df, results_dict, model)
    class_data = analysis_min_ind(df, results_dict, model)
    return class_data

def value_calculation_class(class_data):
    # test significance of results (t-test) on correct and incorrect
    correct = class_data["correct"]
    incorrect = class_data["incorrect"]
    significance, t_stat, p_value, deg_freedom = stat_sign(correct, incorrect, alpha=0.05)
    print("significance: ", significance)
    print("t_stat: ", t_stat)
    print("deg_freedom: ", deg_freedom)
    print("p_value: ", p_value, "alpha: ", 0.05)

def value_calculation_total(total_data1, total_data2):
    # test significance of results (t-test) on two datasets
    significance, t_stat, p_value, deg_freedom = stat_sign(total_data1, total_data2, alpha=0.05)
    print("significance: ", significance)
    print("t_stat: ", t_stat)
    print("deg_freedom: ", deg_freedom)
    print("p_value: ", p_value, "alpha: ", 0.05)

def total_comparison(file1, file2):
    # load data
    df1 = pd.read_json(file1)
    df2 = pd.read_json(file2)

    # make analysis
    _, total_data1 = total_max_ind_lemma(df1)
    _, total_data2 = total_max_ind_lemma(df2)

    # test significance of results (t-test) on two datasets
    value_calculation_total(total_data1, total_data2)

def main():
    # input file, which is nbest_predictions.json, path 
    pred_file = r"C:\Users\pingu\FinQA_replication\dataset\nbest_predictions.json"
    # test file path, (dataset\test.json)
    ori_file = r"C:\Users\pingu\FinQA_replication\dataset\test.json"
    
    # evaluate result and create dictionary of results
    results_dict = classify_rows(pred_file, ori_file)

    # load data
    df = pd.read_json(ori_file)

    #################
    # WMD
    # class_data = WDM_analysis(df, results_dict)
    # class_data = WDM_analysis_min_ind(df, results_dict)

    #################
    # intersection ratio

    # make analysis
    # class_data = analysis(df, results_dict)
    # class_data = analysis_max_ind_lemma(df, results_dict) # uses similarity of lemmas
    # class_data = analysis_min_ind_lemma(df, results_dict) # uses distance of lemmas
    
    #################
    # total test file analysis
    file1 = r"C:\Users\pingu\FinQA_replication\dataset\test.json"
    file2 = r"C:\Users\pingu\FinQA_replication\dataset\test_GPT_rephrase.json"
    # total_comparison(file1, file2)

    #################
    # nr gold in analysis test sets

    # nr of gold_inds in correct incorrect
    # data = nr_gold_inds_analysis(df, results_dict)

    #################
    # input type influence on correct or incorrect
    data = input_type_anal(df, results_dict)

    #################
    # training set analysis
    
    # gold count
    trainset_path = r"C:\Users\pingu\FinQA_replication\dataset\train.json"
    # training_nr_gold_ind(trainset_path)


    # test significance of results (t-test) on correct and incorrect
    # value_calculation_class(class_data)

if __name__ == "__main__":
    main()
