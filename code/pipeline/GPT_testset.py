import openai
import pandas as pd
import os
from copy import deepcopy
import re
import time
temperatue = 0.2
top_p = 0.8

def space_out_punctuation(text: str) -> str:
  # Add space before commas, parenthesis, semicolons, colons, dollar signs, only if they are not already
  # followed by a space.
  text = re.sub(r'(?<!\s)([,\(\);:\$])', r' \1', text)

  # Add space after commas, parenthesis, semicolons, colons, dollar signs, only if they are not already
  # preceded by a space.
  text = re.sub(r'([,\(\);:\$])(?!\s)', r'\1 ', text)

  # Add space before last period in sentence, only if it is not already
  # followed by a space.
  text = re.sub(r'(?<!\s)(\.)$', r' \1', text)

  return text

def fix_numbers(text: str) -> str:
    # remove up to five commas from numbers (e.g. 1,000,000 -> 1000000)
    text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
    text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
    text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
    text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
    text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
    return text

def make_call(gold_sentence, model):
    try:
        comp = openai.Completion.create(engine=model, prompt=gold_sentence, max_tokens=1024, temperature=temperatue, top_p=top_p)
    except:
        print("api error")
        time.sleep(30)
        comp = openai.Completion.create(engine=model, prompt=gold_sentence, max_tokens=1024, temperature=temperatue, top_p=top_p)
    text = comp.choices[0]["text"].split("\n")[-1]
    return text

def create_augmentations(row, augment_pre, augment_post, model):
    # index of first row in post_text
    break_point = len(row['pre_text'])

    # create row to append to df
    new_row = deepcopy(row.to_dict())

    # create augmented questions
    quest_sentence = augment_pre + new_row['qa']['question'] + augment_post
    
    # Make the API call to OpenAI to get augmentations
    new_question = make_call(quest_sentence, model)

    # time.sleep(1)

    # remove empty strings and space out punctuation
    if new_question != "":
        new_question = space_out_punctuation(new_question)
        # sometime commas are inserted in numbers, remove them
        new_question = fix_numbers(new_question)
    else:
        print("empty question")
        # try again
        new_question = make_call(quest_sentence, model)
        if new_question != "":
            # sometime commas are inserted in numbers, remove them
            new_question = fix_numbers(new_question)
            new_question = space_out_punctuation(new_question)
        else:
            print("used old question")
            new_question = row['qa']['question']

    # update question in new row
    new_row['qa']['question'] = new_question

    # make API call to get augmentations of gold_inds
    for key, text in new_row['qa']['gold_inds'].items():
        # parse key
        parsed_key = key.split("_")
        text_index = int(parsed_key[1])

        # skip fix if text is very short or empty
        if len(text.split(" ")) < 4 or text == "":
            continue

        # create sentence to augment and create list of augmentations
        gold_sentence = augment_pre + text + augment_post
        new_gold = make_call(gold_sentence, model)

        # time.sleep(1)

        # remove empty strings and space out punctuation
        if new_gold != "":
            # sometime commas are inserted in numbers, remove them
            new_gold = fix_numbers(new_gold)
            new_gold = space_out_punctuation(new_gold)
        else:
            print("empty gold")
            # try again
            new_gold = make_call(gold_sentence, model)
            if new_gold != "":
                # sometime commas are inserted in numbers, remove them
                new_gold = fix_numbers(new_gold)
                new_gold = space_out_punctuation(new_gold)
            else:
                print("used old gold")
                new_gold = text

        # update gold_ind in new row
        new_row['qa']['gold_inds'][key] = new_gold

        # if text update pre and post text with new gold
        if parsed_key[0] == "text":
            if text_index < break_point:
                row['pre_text'][text_index] = new_gold
            else:
                row['post_text'][text_index - break_point] = new_gold
        
        # tabels are only updated in gold_inds not elsewhere
    
    return new_row

def main():
    # Set the API key
    openai.api_key = "api_key_here"

    # Set the prompt and model
    model = "text-davinci-003"
    # conservative prompt
    # augment_pre = "Write a grammar corrected version of the entire following sentence: '"
    # augment_post = "'. Any ';' that are in the sentence should be left as is."
    
    # aggressive prompt, may want to change top_p and temperature
    augment_pre = "Write a rephrased version of the following sentence: '"
    augment_post = "'. Any ';' that are in the sentence should be left as is."

    # Read in the data
    input_path = r"C:\Users\pingu\FinQA_replication\dataset\test.json"
    df = pd.read_json(input_path)

    ## Remove retriever columns
    df = df.drop(['table_retrieved','text_retrieved','table_retrieved_all','text_retrieved_all', 'table_ori', 'filename'], axis=1)

    # load data from previous step if it exists
    output_path = r"C:\Users\pingu\FinQA_replication\dataset\test_GPT_rephrase.json"
    if os.path.exists(output_path):
        # Load data from previous step
        df_fix = pd.read_json(output_path)
        # last line of log file
        status_index = int(open(r"C:\Users\pingu\FinQA_replication\dataset\test_GPT_rephrase_log.txt", "r").readlines()[-1])
    else:
        # Create new empty dataframe
        df_fix = pd.DataFrame()
        status_index = -1

    print(len(df))
    print(len(df_fix))

    for df_index, row in df.iterrows():
        # skip rows that have already been augmented
        if df_index <= status_index:
            continue

        # remove unnecessary columns
        row['qa'] = {"question": row['qa']["question"], "program": row['qa']["program"], "gold_inds": row['qa']["gold_inds"], "exe_ans": row['qa']["exe_ans"], "program_re": row['qa']["program_re"]}
        
        # create augmentated rows
        new_row = create_augmentations(row, augment_pre, augment_post, model)

        if new_row is not None:
            # add new rows to df_fix
            new_row['id'] = new_row['id'] + "_GPT_rephrased"
            df_fix = pd.concat([df_fix, pd.DataFrame([new_row])], ignore_index=True)
        
        # print progress and save
        if df_index % 10 == 0:
            # save df to json
            df_fix.to_json(output_path, orient='records', indent=4)
            # in log file save (create it if necessary) the last index that was augmented on new line
            with open(r"C:\Users\pingu\FinQA_replication\dataset\test_GPT_rephrase_log.txt", "a+") as f:
                f.write(str(df_index) + "\n")
            print(df_index)

        # time.sleep(60/20)

    print(len(df_fix))
    df_fix.to_json(output_path, orient='records', indent=4)


if __name__ == "__main__":
    main()