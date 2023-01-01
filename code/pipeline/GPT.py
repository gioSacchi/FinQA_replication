import openai
import pandas as pd
import os
# from nltk import word_tokenize
from copy import deepcopy
import re
import time
temperatue = 0.7
top_p = 0.3

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

def create_augmentations(row, n_aug, augment_pre, model):
    # index of first row in post_text
    break_point = len(row['pre_text'])

    total_old = ""
    total_news = ["" for i in range(n_aug)]

    # create list of rows to append to df
    new_rows = [deepcopy(row.to_dict()) for i in range(n_aug)]

    # create augmented questions
    quest_sentence = augment_pre + new_rows[0]['qa']['question'] + "'"
    
    # Make the API call to OpenAI to get augmentations
    try:
        qustion_comp = openai.Completion.create(engine=model, prompt=quest_sentence, max_tokens=1024, temperature=temperatue, top_p=top_p)
    except:
        print("api error")
        time.sleep(30)
        qustion_comp = openai.Completion.create(engine=model, prompt=quest_sentence, max_tokens=1024, temperature=temperatue, top_p=top_p)
    question_list = qustion_comp.choices[0]["text"].split("\n")

    time.sleep(3)

    # remove empty strings, and remove the first 3 characters (1. ) from each string and space out punctuation
    question_list = [space_out_punctuation(quest[3:]) for quest in question_list if quest[3:] != ""]

    # check if too many augmentations
    if len(question_list) > n_aug:
        print("Too many augmentations " + "question")
        print(qustion_comp.choices[0]["text"].split("\n"))

        #detrmine how many extra augmentations there are
        how_many = len(question_list)
        extra = how_many - n_aug
        print("Extra: " + str(extra))

        # loop thorugh augmentations and remove ones that are much shortar than the others
        selected = []
        for i, augmented in enumerate(question_list):
            if len(augmented) < 0.5*len(new_rows[0]['qa']['question'].split(" ")):
                selected.append(i)
                
        # remove the selected augmentations
        question_list = [quest for i, quest in enumerate(question_list) if i not in selected]
        
        # if still too many augmentations, remove the first ones untill there are the right number
        if len(question_list) > n_aug:
            while len(question_list) > n_aug:
                question_list.pop(0)

    # Check if there are enough augmentations
    if len(question_list) < n_aug:
        print("Wrong number of augmentations " + "question")
        if len(question_list) > 0:
            # change n_aug to match and augment_pre to match
            augment_pre = augment_pre.replace(str(n_aug), str(len(question_list)))
            n_aug = len(question_list)
            # update new_rows
            new_rows = new_rows[:n_aug]
            # update total_news
            total_news = total_news[:n_aug]
        else:
            return None

    # update total_old
    total_old += new_rows[0]['qa']['question'] + " "

    # update questions in new rows
    for i in range(n_aug):
        new_quest = question_list[i]
        new_rows[i]['qa']['question'] = new_quest
        
        # update total_news
        total_news[i] += new_quest + " "

    # make API call to get augmentations of gold_inds
    for key, text in new_rows[0]['qa']['gold_inds'].items():
        # parse key
        parsed_key = key.split("_")
        text_index = int(parsed_key[1])

        # do not augment if table
        if parsed_key[0] == "table":
            continue

        # skip augmntation if text is very short or empty
        if len(text.split(" ")) < 4 or text == "":
            continue

        # create sentence to augment and create list of augmentations
        gold_sentence = augment_pre + text + "'"
        try:
            gold_comp = openai.Completion.create(engine=model, prompt=gold_sentence, max_tokens=1024, temperature=temperatue, top_p=top_p)
        except:
            print("api error")
            time.sleep(30)
            gold_comp = openai.Completion.create(engine=model, prompt=gold_sentence, max_tokens=1024, temperature=temperatue, top_p=top_p)
        gold_list = gold_comp.choices[0]["text"].split("\n")

        time.sleep(3)

        # remove empty strings, and remove the first 3 characters (1. ) from each string and space out punctuation
        gold_list = [space_out_punctuation(gold[3:]) for gold in gold_list if gold[3:] != ""]

        # check if too many augmentations
        if len(gold_list) > n_aug:
            print("Too many augmentations " + key)
            print(gold_comp.choices[0]["text"].split("\n"))

            #detrmine how many extra augmentations there are
            how_many = len(gold_list)
            extra = how_many - n_aug
            print("Extra: " + str(extra))

            # loop thorugh augmentations and remove ones that are much shortar than the others
            selected = []
            for i, augmented in enumerate(gold_list):
                if len(augmented) < 0.5*len(text):
                    selected.append(i)
                    
            # remove the selected augmentations
            gold_list = [ind for i, ind in enumerate(gold_list) if i not in selected]
            
            # if still too many augmentations, remove the first ones untill there are the right number
            if len(gold_list) > n_aug:
                while len(gold_list) > n_aug:
                    gold_list.pop(0)

        # Check if there are enough augmentations
        if len(gold_list) < n_aug:
            print("Wrong number of augmentations " + key)
            if len(gold_list) > 0:
                # change n_aug to match and augment_pre to match
                augment_pre = augment_pre.replace(str(n_aug), str(len(question_list)))
                n_aug = len(gold_list)
                # update new_rows
                new_rows = new_rows[:n_aug]
                # update total_news
                total_news = total_news[:n_aug]
            else:
                return None

        # update total_old
        total_old += new_rows[0]['qa']['gold_inds'][key] + " "

        # update gold_ind in new rows
        for i in range(n_aug):
            new_ind = gold_list[i]
            new_rows[i]['qa']['gold_inds'][key] = new_ind

            # update pre and post text with new text
            if text_index < break_point:
                row['pre_text'][text_index] = new_ind
            else:
                row['post_text'][text_index - break_point] = new_ind

            # update total_news
            total_news[i] += new_ind + " "
    
    # check if any of the new rows are the same as the old row and remove them
    remove_indices = []
    for i, total_new in enumerate(total_news):
        if total_new == total_old:
            remove_indices.append(i)
    
    new_rows = [new_rows[i] for i in range(n_aug) if i not in remove_indices]

    return new_rows

def main():
    # Set the API key
    openai.api_key = "api_key"

    # Set the prompt and model
    n_aug = 5
    augment_pre = "Create exactly " + str(n_aug) + " different rephrasings (excluding the sentence) of the following sentence: '"
    model = "text-davinci-003"

    # Read in the data
    input_path = r"C:\Users\pingu\FinQA_replication\dataset\train.json"
    df = pd.read_json(input_path)

    ## Remove retriever columns
    df = df.drop(['table_retrieved','text_retrieved','table_retrieved_all','text_retrieved_all', 'table_ori', 'filename'], axis=1)

    # load data from prvious step if it exists
    output_path = r"C:\Users\pingu\FinQA_replication\dataset\train_GPT_augmented.json"
    if os.path.exists(output_path):
        df_aug = pd.read_json(output_path)
        # last line of log file
        status_index = int(open(r"C:\Users\pingu\FinQA_replication\dataset\train_GPT_augmented_log.txt", "r").readlines()[-1])
    else:
        df_aug = df
        status_index = -1

    print(len(df))
    print(len(df_aug))

    for df_index, row in df.iterrows():
        # skip rows that have already been augmented
        if df_index <= status_index:
            continue

        # remove unnecessary columns
        row['qa'] = {"question": row['qa']["question"], "program": row['qa']["program"], "gold_inds": row['qa']["gold_inds"], "exe_ans": row['qa']["exe_ans"], "program_re": row['qa']["program_re"]}
        
        # create augmentated rows
        new_rows = create_augmentations(row, n_aug, augment_pre, model)

        if new_rows is None:
            continue

        # add new rows to df
        for i, new_row in enumerate(new_rows):
            new_row['id'] = new_row['id'] + "_GPT_" + str(i)
            # df_aug = pd.DataFrame.append(df_aug, new_row, ignore_index=True)
            # update row using new_row and pd.concat
            df_aug = pd.concat([df_aug, pd.DataFrame([new_row])], ignore_index=True)
        
        # print progress and save
        if df_index % 10 == 0:
            print(df_index)
            # save df to json
            df_aug.to_json(output_path, orient='records', indent=4)
            # in log file save (create it if necessary) the last index that was augmented on new line
            with open(r"C:\Users\pingu\FinQA_replication\dataset\train_GPT_augmented_log.txt", "a+") as f:
                f.write(str(df_index) + "\n")

        # add a sleep to avoid rate limit of 20 requests per minute, 60/20 = 3 seconds per request but we make at least 2 requests per row so 6 seconds
        time.sleep((60/20)*2)

    print(len(df_aug))
    df_aug.to_json(output_path, orient='records', indent=4)


if __name__ == "__main__":
    main()