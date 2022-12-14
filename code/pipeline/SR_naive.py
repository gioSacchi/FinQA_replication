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

# random.seed(1)

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
    new_row = deepcopy(row)
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
        threshold =  random.random() # use this instead of uniform 0-1 to increse number of replacements

        # computing n from threshold and choose which to sample
        n = math.ceil(len(words) * threshold)
        sample = random.sample(range(len(words)), n)
        sampled_words = [words[index] for index in sample]
        sampled_tags = [convert_to_wn_pos(tags[index]) for index in sample] if less_naive else [None for _ in sample]

        # lemmatize sampled words
        sampled_lemmas = []
        for word, tag in zip(sampled_words, sampled_tags):
          if tag != None and tag != "":
            sampled_lemmas.append(lemmatizer.lemmatize(word, tag))
          else:
            sampled_lemmas.append(lemmatizer.lemmatize(word))
        # sampled_lemmas = [lemmatizer.lemmatize(word, tag)  else lemmatizer.lemmatize(word) for word, tag in zip(sampled_words, sampled_tags)]

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
            
            # replace word with new_word in text and in words
            new_text = replace_nth_instance(new_text, instance_of_selected, word, new_word)
            words[index] = new_word

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
    threshold =  random.random() 

    # computing n from threshold and choose which to sample
    n = math.ceil(len(words) * threshold)
    sample = random.sample(range(len(words)), n)
    sampled_words = [words[index] for index in sample]
    sampled_tags = [convert_to_wn_pos(tags[index]) for index in sample] if less_naive else [None for _ in sample]

    # lemmatize sampled words
    sampled_lemmas = []
    for word, tag in zip(sampled_words, sampled_tags):
      if tag != None and tag != "":
        sampled_lemmas.append(lemmatizer.lemmatize(word, tag))
      else:
        sampled_lemmas.append(lemmatizer.lemmatize(word))
    # sampled_lemmas = [lemmatizer.lemmatize(word, tag) if (tag != "" and not None) else lemmatizer.lemmatize(word) for word, tag in zip(sampled_words, sampled_tags)]

    new_question = question

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
        
        # replace word with new_word in text and in words
        new_question = replace_nth_instance(new_question, instance_of_selected, word, new_word)
        words[index] = new_question

    # update new_row
    new_row['qa']['question'] = new_question

    total_old += question
    total_new += new_question

    if total_new == total_old:
        # Don't create new question
        return None
    # print(new_row['qa']["gold_inds"])

    return new_row

def augment_number(row, df_index):
  # Make a realy deep copy of the row.
  new_row = deepcopy(row.to_dict())
  qa = new_row['qa']

  program = new_row['qa']['program']
  # Numbers from program using regex, including negatives 
  # and decimals but skipping numbers starting with _ or # (e.g. _100, #1)
  numbers = re.findall(r'(?<![_#])-?\b\d+(?:\.\d+)?\b%?', program)

  ## If there are any duplcate numbers, continue
  if len(numbers) != len(set(numbers)):
    return None

  # Randomly select n numbers from the list
  threshold = random.random()
  n = math.ceil(len(numbers) * threshold)
  random_selected_numbers = random.sample(numbers, n)

  new_program = program

  # Replace the numbers in the program with a random number
  for number in random_selected_numbers:
    if "%" in number:
      new_number = str(random.randint(1, 10000)/100)
    else:
      random_addition = random.randint(int(-abs(float(number) * threshold)), int(abs(float(number) * threshold)))
      new_number = str(round(float(number) + random_addition, 2))
    # Ensuring that no new number coincides with an old number creating attribution problems
    # In reality could look onlu at random_selected_numbers[i:] since numbers before that have already been changed
    if new_number in random_selected_numbers:
      continue

    ## Update program
    new_program = good_replace(new_program, number, new_number)
    new_row['qa']['program'] = new_program

    # index of first row in post_text
    break_point = len(new_row['pre_text'])

    ## Update gold inds and storing modified keys with corresponing new number
    for key, value in new_row['qa']['gold_inds'].items():
      new_gold_ind = good_replace(value, number, new_number)
      
      # Updating gold_ind when sentence has been changed
      if new_gold_ind != value:
        parsed_key = key.split("_")
        row_index = int(parsed_key[1])
        
        # Updating pre or post text and table with new number
        if parsed_key[0] == "table":
          for col_index, table_col in enumerate(new_row['table'][row_index]):
            new_row['table'][row_index][col_index] = good_replace(table_col, number, new_number) 
        else:
          if row_index < break_point:
            new_row['pre_text'][row_index] = good_replace(new_row['pre_text'][row_index], number, new_number)
          else:
            new_row['post_text'][row_index-break_point] = good_replace(new_row['post_text'][row_index-break_point], number, new_number)
      new_row['qa']['gold_inds'][key] = new_gold_ind

    # ## Update pre_text, post_test and table
    # ## TODO: Only update rows present in gold_inds?
    # for index, text_line in enumerate(new_row['pre_text']):
    #   new_row['pre_text'][index] = good_replace(text_line, number, new_number)
    
    # for index, text_line in enumerate(new_row['post_text']):
    #   new_row['post_text'][index] = good_replace(text_line, number, new_number)
    
    # for row_index, table_row in enumerate(new_row['table']):
    #   for col_index, table_col in enumerate(table_row):
    #     new_row['table'][row_index][col_index] = good_replace(table_col, number, new_number) 

  if new_program == program:
    ## Don't create new question
    return None
  
  ## Update exe_ans
  invalid_flag, exe_ans = eval_program(program_tokenization(new_program), new_row['table'])
  if invalid_flag:
    return None
  
  new_row['qa']['exe_ans'] = exe_ans

  return new_row

def main():
  input_path = r"E:\FinQA_replication\dataset\train.json"
  df = pd.read_json(input_path)

  print(len(df))

  # number of generated sentences per original sentence
  num_aug = 5
  random.seed(3)
  ## Remove retriever columns
  df = df.drop(['table_retrieved','text_retrieved','table_retrieved_all','text_retrieved_all', 'table_ori', 'filename'], axis=1)
  # random_ind = random.sample(range(len(df)), 3)
  for df_index, row in df.iterrows():
  # for df_index in random_ind:
    # row = df.iloc[df_index]
    # Dropping unneeded columns, remove program_re???
    row['qa'] = {"question": row['qa']["question"], "program": row['qa']["program"], "gold_inds": row['qa']["gold_inds"], "exe_ans": row['qa']["exe_ans"], "program_re": row['qa']["program_re"]}
    if df_index % 100 == 0:
      print(df_index)
    for n in range(num_aug):
    # for n in range(1):
      new_row = augment_number(row, df_index)
      if new_row:
        new_row_2 = naive_synonym_replacement(new_row, df_index, less_naive=True)
        # print(new_row_2['qa']["gold_inds"])
        # print(row['qa']["gold_inds"])
        if new_row_2:
          new_row_2['id'] = new_row_2['id'] + "_SR_PoS_num_aug_" + str(df_index) + "_" + str(n)
          df = pd.DataFrame.append(df, new_row_2, ignore_index=True)
          # print("New:", new_row_2["qa"]["question"], new_row_2["qa"]["gold_inds"])
          # print("Old:", row["qa"]["question"], row["qa"]["gold_inds"])

  print(len(df))
  output_path = r"E:\FinQA_replication\dataset\train_SR_PoS_augmented.json"
  df.to_json(output_path, orient='records', indent=4)


"""problem in general is that it is hard to get correct form of word. eg changes 
    gives synset change which has lemma alter which will replace changes. But correct 
    would be plural form of alter alterations. How does one do that? Put in to a 
    grammar check. Necessary? Maybe adds noise which is good anyways"""

if __name__ == '__main__':
  main()


# TODO: while loop to ensure