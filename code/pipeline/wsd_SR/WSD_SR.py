import math
import pandas as pd
import random
import nltk
from config import parameters as conf
from WSD import WSD
from copy import deepcopy
import re


# download nltk packages first time
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')
# nltk.download('universal_tagset')   
# nltk.download('stopwords')

from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
# add %, ' and 's to stop words. These cannot b handled by WSD and are not useful for our purposes
stop_words.add("%")
stop_words.add("'s")
stop_words.add("'")

allowed_word_classes_WSD = ['PRT', 'ADV', 'ADJ', 'NOUN', 'VERB']

def tokenize(text):
    # get all words in text
    words = re.findall(r'\b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b', text)
    return words

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

# replace_nth_instance(string, n, word, replacement) replaces the nth instance of 'word' in 'string' with 'replacement'
def replace_nth_instance(text, n, old, new):
    # Create a list of lowercase words in the string
    words = text.lower().split()
    old = old.lower()

    # Replace the nth instance of 'word' in the list with 'replacement'
    # Note: we need to keep track of how many instances of 'word' we have seen so far
    instances_seen = 0
    for i in range(len(words)):
        if words[i] == old:
            instances_seen += 1
            if instances_seen == n:
                words[i] = new
                break
    # Join the words back into a single string and return the result
    return " ".join(words)

def replacement(text, meanings, indecies):
    # iterate through meanings and indecies
    for meaning, index in zip(meanings, indecies):
        words = tokenize(text)
        word = words[index]
        pos = meaning.split(".")[-2]
        lemma = lemmatizer.lemmatize(word, pos=pos)
        
        # get synonyms
        synonyms = get_synonyms(meaning, word, lemma)
        if len(synonyms) == 0:
            continue
        # new word
        new_word = random.choice(synonyms)

        # find all occurences of word in question and their indecies
        occurences = [i for i, w in enumerate(words) if w == word]
        selected_occurence = occurences.index(index) + 1

        # replace word in text
        text = replace_nth_instance(text, selected_occurence, word, new_word)
    return text

def get_synonyms(meaning, word, lemma):
    # get synonyms for a word in a given meaning, remove the word itself and the lemma
    synonyms = set()
    for l in wn.synset(meaning).lemmas():
        synonym = l.name().replace("_", " ").replace("-", " ").lower()
        synonyms.add(synonym)

    if word in synonyms: 
        synonyms.remove(word)
    if lemma in synonyms:
        synonyms.remove(lemma)
    return list(synonyms)

def preprocess_text(row):
    # takes in row and return dictionary with topkenized text, pos tags, lemmas and selected indecies
    processed_row = []
    map_dict = {}

    qa = row['qa']
    id = row['id']

    # break down question into tokens
    question = qa['question']
    question_tokens = tokenize(question)

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
    allowed_indecies = []
    for i, tag in enumerate(question_tags):
        if tag in allowed_word_classes_WSD and question_tokens[i].lower() not in stop_words:
            allowed_indecies.append(i)    
    
    # randomly select n words from eligible words
    threshold = 0.35 + 0.65*random.random()
    n = math.ceil(len(allowed_indecies) * threshold)
    selected_indecies = random.sample(allowed_indecies, n)

    #create dictionary for instace_ids
    instance_ids = {i: id + "_question_" + str(i) for i in selected_indecies}

    question_dict = {"sentence_id": id + "_question" , "words": question_tokens, "lemmas": question_lemmas, "pos_tags": question_tags, "instance_ids": instance_ids}
    processed_row.append(question_dict)
    
    # update map_dict
    map_dict["question"] = [id + "_question_" + str(i) for i in selected_indecies]

    # break down gold_inds into tokens¨
    gold_inds = qa['gold_inds']
    for key, value in gold_inds.items():
        gold_ind_tokens = tokenize(value)
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
        allowed_indecies = []
        for i, tag in enumerate(gold_ind_tags):
            if tag in allowed_word_classes_WSD and gold_ind_tokens[i].lower() not in stop_words:
                allowed_indecies.append(i)
        
        # randomly select n words from eligible words
        threshold = 0.35 + 0.65*random.random()
        n = math.ceil(len(allowed_indecies) * threshold)
        selected_indecies = random.sample(allowed_indecies, n)

        # make instane ids for gold inds
        instance_ids = {i: id + "_" + key + "_" + str(i) for i in selected_indecies}
        gold_ind_dict = {"sentence_id": id + "_" + key, "words": gold_ind_tokens, "lemmas": gold_ind_lemmas, "pos_tags": gold_ind_tags, "instance_ids": instance_ids}
        processed_row.append(gold_ind_dict)

        # update map_dict
        map_dict[key] = [id + "_" + key + "_" + str(i) for i in selected_indecies]

    return processed_row, map_dict # TODO: add n_aug so same rows are different

def create_row(row, key_map, wsd_output):
    # index of first row in post_text
    break_point = len(row['pre_text'])

    total_old = ""
    total_new = ""

    # iterate through key map and replace words
    for key, meaning_ids in key_map.items():
        # get meanings
        meanings = []
        indecies = []
        for meaning_id in meaning_ids:
            # ensuring that all meanings are in wsd_output, takes care of Invalid lemma error
            value = wsd_output.get(meaning_id)
            if value is not None:
                meanings.append(value)
                # get indecies of words to replace
                indecies.append(int(meaning_id.split("_")[-1]))

        # meanings = [wsd_output[meaning_id] for meaning_id in meaning_ids]

        # # get indecies of words to replace
        # indecies = [int(i.split("_")[-1]) for i in meaning_ids]

        # replace words
        if key == "question":
            # get new text
            text = row['qa']['question']
            new_text = replacement(text, meanings, indecies)
            
            # update new row
            row['qa']['question'] = new_text

            # store texts for later
            total_new += new_text + " "
            total_old += text + " "
        else:
            # get new text
            text = row['qa']['gold_inds'][key]
            new_text = replacement(text, meanings, indecies)

            # update new row
            row['qa']['gold_inds'][key] = new_text

            # update pre and post text with new text
            parsed_key = key.split("_")
            text_index = int(parsed_key[1])
            if text_index < break_point:
                row['pre_text'][text_index] = new_text
            else:
                row['post_text'][text_index - break_point] = new_text

            # store texts for later
            total_new += new_text + " "
            total_old += text + " "
    
    if total_new == total_old:
        # Don't create new question
        return None
    
    return row

def main():
    # input_path = r"C:\Users\pingu\FinQA_replication\dataset\train.json"
    df = pd.read_json(conf.train_path)

    print(len(df))

    # number of generated sentences per original sentence
    num_aug = 5
    random.seed(3)

    ## Remove retriever columns
    df = df.drop(['table_retrieved','text_retrieved','table_retrieved_all','text_retrieved_all', 'table_ori', 'filename'], axis=1)

    # big map and big augmenation dict
    big_map = {}
    big_augmentation_list = []

    # first loop to crete big map and big augmentation dict
    # i.e. select all words for syn rep and format for WSD
    for df_index, row in df.iterrows():
        row['qa'] = {"question": row['qa']["question"], "program": row['qa']["program"], "gold_inds": row['qa']["gold_inds"], "exe_ans": row['qa']["exe_ans"], "program_re": row['qa']["program_re"]}
        
        if df_index % 100 == 0:
            print(df_index)
        
        for n in range(num_aug):
            augmentation_list, key_map = preprocess_text(row)

            # add to big map
            big_map[str(df_index) + "_" + str(n)] = key_map
            big_augmentation_list.extend(augmentation_list)
    
    #convert big_augmentation_list to dictionary
    big_augmentation_dict = {i: elem for i, elem in enumerate(big_augmentation_list)}

    # call wsd and get meanings of selected words
    wsd_output = WSD(big_augmentation_dict)

    counter = 0

    # loop through big map and create new rows
    # i.e. replace words with synonym and add new row to dataframe
    for key, key_map in big_map.items():
        # get row
        row_index, n = key.split("_")
        row = df.iloc[int(row_index)]
        new_row = deepcopy(row.to_dict())

        # go though key map and replace words with synonyms from wsd_output
        new_row = create_row(new_row, key_map, wsd_output) # TODO: add check for Invalid lemma

        if new_row:
            new_row['id'] = new_row['id'] + "_WSD_SR_" + str(row_index) + "_" + n
            df = pd.DataFrame.append(df, new_row, ignore_index=True)
        
        counter += 1
        if counter % 100 == 0:
            print(counter)

    print(len(df))
    # output_path = r"C:\Users\pingu\FinQA_replication\dataset\train_WSD_SR_augmented.json"
    df.to_json(conf.model_output, orient='records', indent=4)


"""problem in general is that it is hard to get correct form of word. eg changes 
    gives synset change which has lemma alter which will replace changes. But correct 
    would be plural form of alter alterations. How does one do that? Put in to a 
    grammar check. Necessary? Maybe adds noise which is good anyways"""

if __name__ == '__main__':
    main()