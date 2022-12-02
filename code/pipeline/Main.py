import math
import random
import pandas as pd
import time
import json
import re
from copy import deepcopy
from utils import *

def main():
  input_path = r"C:\Users\alexa\projects\su-kex\FinQA_replication\dataset\train.json"
  df = pd.read_json(input_path)

  print(len(df))

  ## Remove retriever columns
  df = df.drop(['table_retrieved','text_retrieved','table_retrieved_all','text_retrieved_all', 'table_ori', 'filename'], axis=1)

  for df_index, row in df.iterrows():
    # Dropping unneeded columns, remove program_re???
    row['qa'] = {"question": row['qa']["question"], "program": row['qa']["program"], "gold_inds": row['qa']["gold_inds"], "exe_ans": row['qa']["exe_ans"], "program_re": row['qa']["program_re"]}
  
    row = augment_number(row, df_index)
    if row:
      df = pd.DataFrame.append(df, row, ignore_index=True)

  print(len(df))
  output_path = r"C:\Users\alexa\projects\su-kex\FinQA_replication\code\pipeline\output\train_augmented.json"
  df.to_json(output_path, orient='records', indent=4)



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


    ## Update program
    new_program = good_replace(new_program, number, new_number)
    new_row['qa']['program'] = new_program

    ## Update gold inds
    for key, value in new_row['qa']['gold_inds'].items():
      new_row['qa']['gold_inds'][key] = good_replace(value, number, new_number)

    ## Update pre_text, post_test and table
    ## TODO: Only update rows present in gold_inds?
    for index, text_line in enumerate(new_row['pre_text']):
      new_row['pre_text'][index] = good_replace(text_line, number, new_number)
    
    for index, text_line in enumerate(new_row['post_text']):
      new_row['post_text'][index] = good_replace(text_line, number, new_number)
    
    for row_index, table_row in enumerate(new_row['table']):
      for col_index, table_col in enumerate(table_row):
        new_row['table'][row_index][col_index] = good_replace(table_col, number, new_number) 

  if new_program == program:
    ## Don't create new question
    return None
  
  ## Update exe_ans
  invalid_flag, exe_ans = eval_program(program_tokenization(new_program), new_row['table'])
  if invalid_flag:
    return None
  
  new_row['qa']['exe_ans'] = exe_ans
  
  ## Update id TODO: Move to main function
  new_row['id'] = new_row['id'] + "_" + "augmented_" + str(df_index)
  return new_row


  
if __name__ == '__main__':
  main()
    