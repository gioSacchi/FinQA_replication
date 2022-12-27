import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

stop_words = set(stopwords.words('english'))

# Calculate the Word Mover's Distance between two sentences
def compare_sentences(sentence1, sentence2):
    # Tokenize the sentences
    sentence1_tokens = word_tokenize(sentence1)
    sentence2_tokens = word_tokenize(sentence2)

    # Remove the stop words
    sentence1_tokens = [token for token in sentence1_tokens if token not in stop_words]
    sentence2_tokens = [token for token in sentence2_tokens if token not in stop_words]
    
    # Load the model
    model = Word2Vec.load("word2vec.model")

    # Calculate the Word Mover's Distance
    distance = model.wmdistance(sentence1_tokens, sentence2_tokens)

    # Print the distance
    print("Word Mover's Distance:", distance)

# Compare two sentences
compare_sentences("The cat sat on the mat.", "The cat lay on the carpet.")