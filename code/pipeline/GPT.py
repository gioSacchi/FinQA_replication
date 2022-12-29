import openai
import pandas as pd
# from nltk import word_tokenize
from copy import deepcopy
import re
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
    quest_sentence = augment_pre + new_rows[0]['qa']['question']
    
    # Make the API call to OpenAI to get augmentations
    qustion_comp = openai.Completion.create(engine=model, prompt=quest_sentence, max_tokens=1024, temperature=temperatue, top_p=top_p)
    question_list = qustion_comp.choices[0]["text"].split("\n")

    # remove empty strings, and remove the first 3 characters (1. ) from each string and space out punctuation
    question_list = [space_out_punctuation(quest[3:]) for quest in question_list if quest != ""]

    # Check if there are enough augmentations
    if len(question_list) != n_aug:
        print("Wrong number of augmentations " + "question")
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

        # create sentence to augment and create list of augmentations
        gold_sentence = augment_pre + text
        gold_comp = openai.Completion.create(engine=model, prompt=gold_sentence, max_tokens=1024, temperature=temperatue, top_p=top_p)
        gold_list = gold_comp.choices[0]["text"].split("\n")

        # remove empty strings, and remove the first 3 characters (1. ) from each string and space out punctuation
        gold_list = [space_out_punctuation(gold[3:]) for gold in gold_list if gold != ""]

        # Check if there are enough augmentations
        if len(gold_list) != n_aug:
            print("Wrong number of augmentations " + key)
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
    openai.api_key = "add your key here"

    # Set the prompt and model
    n_aug = 5
    augment_pre = "Create exactly " + str(n_aug) + " different rephrasings (excluding the sentence) of the following sentence: "
    model = "text-davinci-003"

    input_path = r"C:\Users\pingu\FinQA_replication\dataset\train.json"
    df = pd.read_json(input_path)

    print(len(df))

    ## Remove retriever columns
    df = df.drop(['table_retrieved','text_retrieved','table_retrieved_all','text_retrieved_all', 'table_ori', 'filename'], axis=1)

    for df_index, row in df.iterrows():
        # remove unnecessary columns
        row['qa'] = {"question": row['qa']["question"], "program": row['qa']["program"], "gold_inds": row['qa']["gold_inds"], "exe_ans": row['qa']["exe_ans"], "program_re": row['qa']["program_re"]}
        
        if df_index % 100 == 0:
            print(df_index)
        
        # create augmentated rows
        new_rows = create_augmentations(row, n_aug, augment_pre, model)

        if new_rows is None:
            continue

        # add new rows to df
        for i, new_row in enumerate(new_rows):
            new_row['id'] = new_row['id'] + "_GPT_" + str(i)
            df = pd.DataFrame.append(df, new_row, ignore_index=True)

    print(len(df))
    output_path = r"C:\Users\pingu\FinQA_replication\dataset\train_GPT_augmented.json"
    df.to_json(output_path, orient='records', indent=4)


if __name__ == "__main__":
    main()