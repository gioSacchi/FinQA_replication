import openai
import pandas as pd
from nltk import word_tokenize
from copy import deepcopy
temperatue = 0.7
top_p = 0.3

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

    # update questions in new rows
    for i in range(n_aug):
        new_quest = question_list[i]
        # TODO: add question processing here
        new_rows[i]['qa']['question'] = new_quest
        
        # update total_news
        total_news[i] += new_quest + " "
    
    # update total_old
    total_old += new_rows[0]['qa']['question'] + " "

    # make API call to get augmentations of gold_inds
    for key, text in new_rows[0]['qa']['gold_inds'].items():
        gold_sentence = augment_pre + text
        gold_comp = openai.Completion.create(engine=model, prompt=gold_sentence, max_tokens=1024, temperature=temperatue, top_p=top_p)
        gold_list = gold_comp.choices[0]["text"].split("\n")

        # update gold_ind in new rows
        for i in range(n_aug):
            new_ind = gold_list[i]
            # TODO: add question processing here
            new_rows[i]['qa']['gold_inds'][key] = new_ind

            # update pre and post text with new text
            parsed_key = key.split("_")
            text_index = int(parsed_key[1])
            if text_index < break_point:
                row['pre_text'][text_index] = new_ind
            else:
                row['post_text'][text_index - break_point] = new_ind

            # update total_news
            total_news[i] += new_ind + " "
        
        # update total_old
        total_old += new_rows[0]['qa']['gold_inds'][key] + " "
    
    # check if any of the new rows are the same as the old row and remove them
    for i, total_new in enumerate(total_news):
        if total_new == total_old:
            new_rows.pop(i)

    return new_rows

def main():
    # Set the API key
    openai.api_key = "add_key_here"

    # Set the prompt and model
    n_aug = 5
    augment_pre = "Create " + str(n_aug) + " different rephrasings of the following sentence: "
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

        # add new rows to df
        for i, new_row in enumerate(new_rows):
            new_row['id'] = new_row['id'] + "_GPT_" + str(i)
            df = pd.DataFrame.append(df, new_row, ignore_index=True)

    print(len(df))
    output_path = r"C:\Users\pingu\FinQA_replication\dataset\train_WSD_SR_augmented.json"
    df.to_json(output_path, orient='records', indent=4)


if __name__ == "__main__":
    main()