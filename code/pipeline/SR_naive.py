## using https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py#L4 as reference for naive approach

import random
import re
from copy import deepcopy
import math
from utils import *
import pandas as pd

## for the first time you use wordnet
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords

random.seed(1)

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

def get_synonyms_naive(word, lemma, tag = None, less_naive = False):
    # make a list of all the synonyms (lemmas) to word 
    # from all of it's possible meanings (all wordnet.synsets)
    synonyms = set()
    for syn in wn.synsets(word, pos=tag): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            # synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm']) # sub with regex, Why do this??
            ###### thousends -> 1000 , which becomes "". Is it ok to have numbers?
            synonyms.add(synonym) 
    if word in synonyms: 
        synonyms.remove(word)
    if lemma in synonyms:
        synonyms.remove(lemma)
    return list(synonyms)

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

    # to check if any changes have been made
    total_old = ""
    total_new = ""

    lemmatizer = WordNetLemmatizer()

    # update gold_inds and corresponding texts or tabels with new synonyms
    for key, text in gold_inds.items():
        # skip if gold_ind is a table
        if "table" in key:
            continue

        # get all words in text
        words = re.findall(r'\b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b', text)

        # sort out words that are in stop words
        words = [word for word in words if word not in stop_words]

        # get tags for words
        if less_naive:
          tags = []
          for _, tag in pos_tag(words):
            tags.append(tag)

        # randomly setting threshold
        threshold = random.random() 

        # computing n from threshold and choose which to sample
        n = math.ceil(len(words) * threshold)
        sample = random.sample(range(len(words)), n)
        sampled_words = [words[index] for index in sample]
        sampled_tags = [convert_to_wn_pos(tags[index]) for index in sample] if less_naive else [None for _ in sample]

        # lemmatize sampled words
        sampled_lemmas = [lemmatizer.lemmatize(word, tag) for word, tag in zip(sampled_words, sampled_tags)]
        lemma_text = " ".join(sampled_lemmas)

        new_text = text

        for i, word in enumerate(sampled_words):
            # get index and lemma of word
            index = sample[i]
            lemma = sampled_lemmas[i]

            # find alla instances of word in words and compute wich of them index corresponds to
            all_indices = [i for i, x in enumerate(words) if x == word]
            instance_of_selected = all_indices.index(index) + 1

            # get synonyms
            tag = sampled_tags[i]
            synonyms = get_synonyms_naive(word, lemma, tag, less_naive)
            if len(synonyms) == 0:
              continue
            
            # choose synonym
            new_word = random.choice(synonyms)
            # check that new word is not in sampled words (to avoid duplicates). To avoid cases
            # such as new_word = "one year" and one of the two being in sampled words
            if new_word in lemma_text:
              continue
            
            # replace word with new_word in text
            new_text = replace_nth_instance(new_text, instance_of_selected, word, new_word)

        # update new_row
        new_row['qa']['gold_inds'][key] = new_text

        # update pre and post text with new text
        parsed_key = key.split("_")
        row_index = int(parsed_key[1])
        if row_index < break_point:
          new_row['pre_text'][row_index] = new_text
        else:
          new_row['post_text'][row_index - break_point] = new_text

        total_old += text + " "
        total_new += new_text + " "

    # update question with synonyms
    # get all words in text
    words = re.findall(r'\b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b', question)

    # sort out words that are in stop words
    words = [word for word in words if word not in stop_words]

    # get tags for words
    if less_naive:
      tags = []
      for _, tag in pos_tag(words):
        tags.append(tag)

    # randomly setting threshold
    threshold = random.random() 

    # computing n from threshold and choose which to sample
    n = math.ceil(len(words) * threshold)
    sample = random.sample(range(len(words)), n)
    sampled_words = [words[index] for index in sample]
    sampled_tags = [tags[index] for index in sample] if less_naive else [None for _ in sample]

    # lemmatize sampled words
    sampled_lemmas = [lemmatizer.lemmatize(word, convert_to_wn_pos(tag)) for word, tag in zip(sampled_words, sampled_tags)]
    lemma_text = " ".join(sampled_lemmas)

    new_question = question

    for i, word in enumerate(sampled_words):
        # get index and lemma of word
        index = sample[i]
        lemma = sampled_lemmas[i]

        # find alla instances of word in words and compute wich of them index corresponds to
        all_indices = [i for i, x in enumerate(words) if x == word]
        instance_of_selected = all_indices.index(index) + 1

        # get synonyms
        tag = tags[index] if less_naive else None
        synonyms = get_synonyms_naive(word, lemma, tag, less_naive)
        if len(synonyms) == 0:
          continue
        
        # choose synonym
        new_word = random.choice(synonyms)
        # check that new word is not in sampled words (to avoid duplicates)
        if new_word in lemma_text:
          continue
        
        # replace word with new_word in text
        new_question = replace_nth_instance(new_question, instance_of_selected, word, new_word)

    # update new_row
    new_row['qa']['question'] = new_question

    total_old += question
    total_new += new_question

    if total_new == total_old:
        # Don't create new question
        return None

    ## Update id TODO: Move to main function
    new_row['id'] = new_row['id'] + "_augmented_" + str(df_index)

    return new_row

def main():
  input_path = r"E:\FinQA_replication\dataset\train.json"
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