import math
import pandas as pd
import random
import nltk

#download nltk packages first time
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('universal_tagset')   
nltk.download('stopwords')

from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
allowed_word_classes_WSD = ['PRT', 'ADV', 'ADJ', 'NOUN', 'VERB']


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

def preprocess_text(row):
    # takes in row and return dictionary with topkenized text, pos tags, lemmas and selected indecies
    processed_row = []
    map_dict = {}

    qa = row['qa']
    id = row['id']

    # break down question into tokens
    question = qa['question']
    question_tokens = word_tokenize(question)

    #words + tags for lemmatisation
    lemma_tags = pos_tag(question_tokens)

    #tags for WSD
    question_tags = [tag for _, tag in nltk.pos_tag(question_tokens, tagset='universal')]
    question_lemmas = []

    # lemmatize question
    for word, tag in lemma_tags:
        tag = convert_to_wn_pos(tag)
        if tag != None and tag != "":
            question_lemmas.append(lemmatizer.lemmatize(word, tag))
        else:
            question_lemmas.append(lemmatizer.lemmatize(word))

    # determine indicies of words eligible for selection for WSD
    # only select words that are not stop words and are in allowed word classes
    selected_indecies = []
    for i, tag in enumerate(question_tags):
        if tag in allowed_word_classes_WSD and question_tokens[i].lower() not in stop_words:
            selected_indecies.append(i)    
    
    # randomly select n words from eligible words
    threshold = random.random()
    n = math.ceil(len(selected_indecies) * threshold)
    selected_indecies = random.sample(selected_indecies, n)

    #create dictionary for instace_ids
    instance_ids = {i: id + "_question_" + str(i) for i in selected_indecies}

    question_dict = {"id": id + "_question" , "words": question_tokens, "lemmas": question_lemmas, "pos_tags": question_tags, "instance_ids": instance_ids}
    processed_row.append(question_dict)
    
    # update map_dict
    map_dict["question"] = [id + "_question_" + str(i) for i in selected_indecies]

    # break down gold_inds into tokens¨
    gold_inds = qa['gold_inds']
    for key, value in gold_inds.items():
        gold_ind_tokens = word_tokenize(value)
        gold_ind_lemmas = []
        gold_ind_tags = [tag for _, tag in nltk.pos_tag(gold_ind_tokens, tagset='universal')]
        lemma_tags = pos_tag(gold_ind_tokens)

        # lemmatize gold_ind
        for word, tag in lemma_tags:
            tag = convert_to_wn_pos(tag)
            if tag != None and tag != "":
                gold_ind_lemmas.append(lemmatizer.lemmatize(word, tag))
            else:
                gold_ind_lemmas.append(lemmatizer.lemmatize(word))
        
        # determine indicies of words eligible for selection for WSD
        # only select words that are not stop words and are in allowed word classes
        selected_indecies = []
        for i, tag in enumerate(gold_ind_tags):
            if tag in allowed_word_classes_WSD and gold_ind_tokens[i].lower() not in stop_words:
                selected_indecies.append(i)
        
        # randomly select n words from eligible words
        threshold = random.random()
        n = math.ceil(len(selected_indecies) * threshold)
        selected_indecies = random.sample(selected_indecies, n)

        # make instane ids for gold inds
        instance_ids = {i: id + "_" + key + "_" + str(i) for i in selected_indecies}
        gold_ind_dict = {"id": id + "_" + key, "words": gold_ind_tokens, "lemmas": gold_ind_lemmas, "pos_tags": gold_ind_tags, "instance_ids": instance_ids}
        processed_row.append(gold_ind_dict)

        # update map_dict
        map_dict[key] = [id + "_" + key + "_" + str(i) for i in selected_indecies]

    return processed_row, map_dict

def main():
    input_path = r"E:\FinQA_replication\dataset\train.json"
    df = pd.read_json(input_path)

    print(len(df))

    # number of generated sentences per original sentence
    num_aug = 5
    random.seed(3)

    ## Remove retriever columns
    df = df.drop(['table_retrieved','text_retrieved','table_retrieved_all','text_retrieved_all', 'table_ori', 'filename'], axis=1)

    # big map and big augmenation dict
    big_map = {}
    big_augmentation_dict = {}

    # first loop to crete big map and big augmentation dict
    # i.e. select all words for syn rep and format for WSD
    for df_index, row in df.iterrows():
        row['qa'] = {"question": row['qa']["question"], "program": row['qa']["program"], "gold_inds": row['qa']["gold_inds"], "exe_ans": row['qa']["exe_ans"], "program_re": row['qa']["program_re"]}
        
        if df_index % 100 == 0:
            print(df_index)
        
        for n in range(num_aug):
            augmentation_list, key_map = preprocess_text(row)

            # add to big map
            big_map[row['id'] + "_WSD_aug_" + str(df_index) + "_" + str(n)]
            # for elem in augmentation_list:
            #     big_augmentation_dict[row['id'] + "_WSD_aug_" + str(df_index) + "_" + str(n)] = 
            # new_row = WSD_synonym_replacement(row, df_index)
            # if new_row:
            #   new_row['id'] = new_row['id'] + "_SR_PoS_num_aug_" + str(df_index) + "_" + str(n)
            #   df = pd.DataFrame.append(df, new_row, ignore_index=True)

    print(len(df))
    output_path = r"E:\FinQA_replication\dataset\train_SR_PoS_augmented.json"
    df.to_json(output_path, orient='records', indent=4)


"""problem in general is that it is hard to get correct form of word. eg changes 
    gives synset change which has lemma alter which will replace changes. But correct 
    would be plural form of alter alterations. How does one do that? Put in to a 
    grammar check. Necessary? Maybe adds noise which is good anyways"""

if __name__ == '__main__':
  main()