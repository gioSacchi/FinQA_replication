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

    # update gold_inds and corresponding texts or tabels with new synonyms
    for key, text in gold_inds.items():
        # get all words in text
        words = re.findall(r'\b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b', text)

        # sort out words that are in stop words
        words = [word for word in words if word not in stop_words]

        # get tags for words
        if less_naive:
            words = pos_tag(words)

        # randomly setting threshold
        threshold = random.random() 

        # computing n from threshold and choose which to sample
        n = math.ceil(len(words) * threshold)
        sample = random.sample(range(len(words)), n)

        for index in sample:
            word = words[index]
            if less_naive:
                word = word[0]
                tag = word[1]
            # find alla instances of word in words and compute wich of them index corresponds to
            all_indices = [i for i, x in enumerate(words) if x == word]
            instance_of_selected = all_indices.index(index) + 1


        new_text, changes = new_text_naive(text, threshold, less_naive)
        new_row['qa']['gold_inds'][key] = new_text

        total_old += text + " "
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