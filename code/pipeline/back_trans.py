import re
import pandas as pd
from copy import deepcopy
from utils import *
import six
from google.cloud import translate_v2 as translate


def translate_text(source: str, target: str, text: str) -> str:
  """Translates text into the target language.

  Target must be an ISO 639-1 language code.
  See https://g.co/cloud/translate/v2/translate-reference#supported_languages
  """
  translate_client = translate.Client()

  if isinstance(text, six.binary_type):
    text = text.decode("utf-8")

  # Text can also be a sequence of strings, in which case this method
  # will return a sequence of results for each text.
  result = translate_client.translate(
      text, target_language=target, source_language=source)

  print(u"Text: {}".format(result["input"]))
  print(u"Translation: {}".format(result["translatedText"]))

  return result["translatedText"]


def back_translate_text(text: str, source: str, target: str) -> str:
  translated_text = translate_text(source, target, text)
  back_translated_text = translate_text(target, source, translated_text)

  back_translated_text = space_out_punctuation(back_translated_text)

  print(text)
  print(back_translated_text)

  return back_translated_text


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


def back_translation_augment(row: pd.Series, df_index: int) -> pd.Series:
  print(space_out_punctuation("Hello, (wor!ld) $100.00."))

  # Make a realy deep copy of the row.
  new_row = deepcopy(row.to_dict())

  # extract the question from row
  question = row['qa']['question']

  transit_language = "zh"

  # translate the question to french from english
  translated_question = back_translate_text(
      str(question), "en", transit_language)

  new_row['qa']['question'] = translated_question

  for i, key in enumerate(new_row['qa']['gold_inds'].keys()):
    input_text = new_row['qa']['gold_inds'][key]

    loc_split = key.split("_")
    index = int(loc_split[1])

    if loc_split[0] == "text":
      translated_text = back_translate_text(
          input_text, 'en', transit_language)

      new_row['qa']['gold_inds'][key] = translated_text

      if index < len(new_row['pre_text']):
        new_row['pre_text'][index] = translated_text
      else:
        new_row['post_text'][index -
                             len(new_row['pre_text'])] = translated_text

  new_row['id'] = new_row['id'] + "_" + "augmented_" + str(df_index)

  return pd.Series(new_row)
