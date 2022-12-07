## using https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py#L4 as reference for naive approach

import random
import re
from copy import deepcopy
import math
from utils import *
from operator import itemgetter

import pandas as pd
## for the first time you use wordnet
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.corpus import stopwords



random.seed(1)

stop_words = set(stopwords.words('english'))
# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

def get_synonyms_naive(word):
    # make a list of all the synonyms (lemmas) to word 
    # from all of it's possible meanings (all wordnet.synsets)
    synonyms = set()
    for syn in wn.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            print(synonym)
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm']) # sub with regex, Why do this??
            ###### thousends -> 1000 , which becomes "". Is it ok to have numbers?
            synonyms.add(synonym) 
    if word in synonyms: 
        synonyms.remove(word)
    return list(synonyms)

def get_synonyms_less_naive(word_w_tag):
    word, tag = word_w_tag
    # make a list of all the synonyms (lemmas) to word 
    # from all of it's possible meanings (all wordnet.synsets)
    synonyms = set()
    for syn in wn.synsets(word, pos=convert_to_wn_pos(tag)): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def convert_to_wn_pos(pos):
    if pos.startswith("J"):
        return wn.ADJ
    elif pos.startswith("V"):
        return wn.VERB
    elif pos.startswith("N"):
        return wn.NOUN
    elif pos.startswith("R"):
        return wn.ADV
    else:
        return "" 

def new_text_naive(text: str, threshold: float, less_naive = False):
    # find all words
    words = re.findall(r'\b[^\d\W]+\b', text) # taking into account ' as in don't??? This maybe works \b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b

    new_text = text
    changes = {}

    # find unique words not included in stop_words
    if less_naive:
        word_list = []
        words_with_tags = pos_tag(words)
        for word, tag in words_with_tags:
            if word not in stop_words and word not in list(map(itemgetter(0), word_list)):
                 # do we want to add same word if different tag? If so substitution should be tag based too
                word_list.append((word, tag))
    else:
        word_list = list(set([word for word in words if word not in stop_words]))

    # computing n from threshold and words and sample
    n = math.ceil(len(word_list) * threshold)
    random_word_list = random.sample(word_list, n)

    # Replacing all occurences of selected words with random synonym.
    for random_word in random_word_list:
        # Random_word will be tupple if less naive
        synonyms = get_synonyms_naive(random_word) if not less_naive else get_synonyms_less_naive(random_word)
        if len(synonyms) == 0:
            continue
        new_word = random.choice(synonyms)
        if new_word in random_word_list:    
            # here too should be necessary to look at random_word_list[i:]
            continue
        # print("replaced", random_word, "with", synonym)
        new_text = good_replace(new_text, random_word, new_word)
        changes[random_word] = new_word

    return new_text, changes if changes else None

def naive_synonym_replacement(row, df_index, less_naive = False):
    # copy of row
    new_row = deepcopy(row.to_dict())
    qa = new_row['qa']

    # taking out the question and gold_inds
    # do we want to change with questions too?
    question = qa["question"]
    gold_inds = qa["gold_inds"]

    # index of first row in post_text
    break_point = len(new_row['pre_text'])

    # randomly setting threshold
    threshold = random.random() # maybe one per threshold per thing?

    # to check if any changes have been made
    total_old = ""
    total_new = ""

    # update gold_inds and corresponding texts or tabels with new synonyms
    for key, value in gold_inds.items():
        new_text, changes = new_text_naive(value, threshold, less_naive)
        new_row['qa']['gold_inds'][key] = new_text

        total_old += value + " "
        total_new += new_text + " "

        if changes == None:
            # no need to update anything
            continue

        # Updating pre or post text and table with new synonym
        parsed_key = key.split("_")
        row_index = int(parsed_key[1])
        if parsed_key[0] == "table":
          for col_index, table_col in enumerate(new_row['table'][row_index]):
            new_row['table'][row_index][col_index] = sequence_good_replace(table_col, changes) 
        else:
          if row_index < break_point:
            new_row['pre_text'][row_index] = sequence_good_replace(new_row['pre_text'][row_index], changes)
          else:
            new_row['post_text'][row_index-break_point] = sequence_good_replace(new_row['post_text'][row_index-break_point], changes)

    # update question with synonyms
    new_question, changes = new_text_naive(question, threshold, less_naive)
    new_row["qa"]["question"] = new_question
    total_old += question
    total_new += new_question

    if total_new == total_old:
        # Don't create new question
        return None

    ## Update id TODO: Move to main function
    new_row['id'] = new_row['id'] + "_augmented_" + str(df_index)

    return new_row

def synonym_replacement_WSD(row, df_index):
    # copy of row
    new_row = deepcopy(row.to_dict())
    qa = new_row['qa']

    # taking out the question and gold_inds
    # do we want to change with questions too?
    question = qa["question"]
    gold_inds = qa["gold_inds"]

    # index of first row in post_text
    break_point = len(new_row['pre_text'])

    # randomly setting threshold
    threshold = random.random() # maybe one per threshold per thing?

    # to check if any changes have been made
    total_old = ""
    total_new = ""

    # update gold_inds and corresponding texts or tabels with new synonyms
    for key, value in gold_inds.items():
        new_text, changes = new_text_naive(value, threshold)
        new_row['qa']['gold_inds'][key] = new_text

        total_old += value + " "
        total_new += new_text + " "

        if changes == None:
            # no need to update anything
            continue

        # Updating pre or post text and table with new synonym
        parsed_key = key.split("_")
        row_index = int(parsed_key[1])
        if parsed_key[0] == "table":
          for col_index, table_col in enumerate(new_row['table'][row_index]):
            new_row['table'][row_index][col_index] = sequence_good_replace(table_col, changes) 
        else:
          if row_index < break_point:
            new_row['pre_text'][row_index] = sequence_good_replace(new_row['pre_text'][row_index], changes)
          else:
            new_row['post_text'][row_index-break_point] = sequence_good_replace(new_row['post_text'][row_index-break_point], changes)

    # update question with synonyms
    new_question, changes = new_text_naive(question, threshold)
    new_row["qa"]["question"] = new_question
    total_old += question
    total_new += new_question

    if total_new == total_old:
        # Don't create new question
        return None

    ## Update id TODO: Move to main function
    new_row['id'] = new_row['id'] + "_augmented_" + str(df_index)

    return new_row

def WSD_new_text(text, threshold):
    # find all words
    words = re.findall(r'\b[^\d\W]+\b', text) # taking into account ' as in don't??? This maybe works \b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b

    new_text = text
    changes = {}

    # find unique words not included in stop_words
    word_list = []
    # This tagset is used in the WSD model, that's why we use it
    words_with_tags = pos_tag(words,  tagset='universal')    
    for word, tag in words_with_tags:
        if word not in stop_words and word not in list(map(itemgetter(0), word_list)):
            # do we want to add same word if different tag? If so substitution should be tag based too
            word_list.append((word, tag))

    # computing n from threshold and words and sample
    n = math.ceil(len(word_list) * threshold)
    random_word_list = random.sample(word_list, n)

    # Replacing all occurences of selected words with random synonym.
    for random_word in random_word_list:
        synonyms = get_WSD_synonyms(random_word)
        if len(synonyms) == 0:
            continue
        new_word = random.choice(synonyms)
        if new_word in random_word_list:    
            # here too should be necessary to look at random_word_list[i:]
            continue
        # print("replaced", random_word, "with", synonym)
        new_text = good_replace(new_text, random_word, new_word)
        changes[random_word] = new_word

    return new_text, changes if changes else None

def get_WSD_synonyms(words_w_tags, target):
    word, tag = target
    # make a list of all the synonyms (lemmas) to word 
    # from all of it's possible meanings (all wordnet.synsets)
    synonyms = set()
    meaning = WSD(words_w_tags, word, tag)
    syn = wn.synset(meaning)
    for l in syn.lemmas(): 
        synonym = l.name().replace("_", " ").replace("-", " ").lower()
        synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
        synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def WSD(words_w_tags, word, tag):
    """performes wsd using model
        look at code and bring in right stuff"""
    pass

def main():
  input_path = r"C:\Users\pingu\FinQA_replication\dataset\train.json"
  df = pd.read_json(input_path)

  print(len(df))

  ## Remove retriever columns
  df = df.drop(['table_retrieved','text_retrieved','table_retrieved_all','text_retrieved_all', 'table_ori', 'filename'], axis=1)

  for df_index, row in df.iterrows():
    # Dropping unneeded columns, remove program_re???
    row['qa'] = {"question": row['qa']["question"], "program": row['qa']["program"], "gold_inds": row['qa']["gold_inds"], "exe_ans": row['qa']["exe_ans"], "program_re": row['qa']["program_re"]}
  
    row = naive_synonym_replacement(row, df_index, True)
    if row:
      df = pd.DataFrame.append(df, row, ignore_index=True)

  print(len(df))
  output_path = r"C:\Users\alexa\projects\su-kex\FinQA_replication\code\pipeline\output\train_augmented.json"
  df.to_json(output_path, orient='records', indent=4)


"""problem in general is that it is hard to get correct form of word. eg changes 
    gives synset change which has lemma alter which will replace changes. But correct 
    would be plural form of alter alterations. How does one do that? Put in to a 
    grammar check. Necessary? Maybe adds noise which is good anyways"""

if __name__ == '__main__':
  main()
    