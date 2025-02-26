import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("stopwords")

# Initialize components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def process_text(sentence):
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    words = [w for w in words if w.isalnum()]  # Remove punctuation

    # POS tagging
    tagged = pos_tag(words)

    # Lemmatization and stopword filtering
    processed_words = []
    for word, tag in tagged:
        pos = "n"  # Default to noun
        if tag.startswith("V"):
            pos = "v"  # Verb
        elif tag.startswith("J"):
            pos = "a"  

        lemmatized_word = lemmatizer.lemmatize(word, pos)


        if word not in stop_words or word in ["is", "are", "was", "were", "be", "to", "i", "you", "we", "he", "she", "they", "will"]:
            processed_words.append(lemmatized_word)
            

    doc = nlp(" ".join(processed_words))
    final_sentence = " ".join([token.text for token in doc])

  
    final_sentence = final_sentence.capitalize() + "."

    return final_sentence

# Example usage
if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    keywords = process_text(sentence)
    print("Extracted Keywords:", keywords)





# import nltk
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords, wordnet
# from nltk.stem import WordNetLemmatizer
# from nltk.tag import pos_tag

# # Download necessary NLTK resources
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")
# nltk.download("stopwords")
# nltk.download("omw-1.4")  # Open Multilingual WordNet for translations

# # Initialize components
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words("english"))

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# def get_synonyms(word):
#     """Fetch synonyms from WordNet."""
#     synonyms = set()
#     for syn in wordnet.synsets(word):
#         for lemma in syn.lemmas():
#             synonyms.add(lemma.name())
#     return list(synonyms)[:3]  # Return top 3 synonyms

# def process_text(sentence):
#     sentence = sentence.lower()
#     words = word_tokenize(sentence)
#     words = [w for w in words if w.isalnum()]  # Remove punctuation

#     # POS tagging
#     tagged = pos_tag(words)

#     # Lemmatization and stopword filtering
#     processed_words = []
#     for word, tag in tagged:
#         pos = "n"  # Default to noun
#         if tag.startswith("V"):
#             pos = "v"  # Verb
#         elif tag.startswith("J"):
#             pos = "a"  # Adjective

#         lemmatized_word = lemmatizer.lemmatize(word, pos)

#         # Keep important words
#         if word not in stop_words or word in ["is", "are", "was", "were", "be", "to", "i", "you", "we", "he", "she", "they", "will"]:
#             processed_words.append(lemmatized_word)

#     # Reconstruct sentence using spaCy
#     doc = nlp(" ".join(processed_words))
#     final_sentence = " ".join([token.text for token in doc])

#     # Fetch synonyms using OMW 1.4
#     synonyms_dict = {word: get_synonyms(word) for word in processed_words}

#     # Capitalize first letter and add a period
#     final_sentence = final_sentence.capitalize() + "."

#     return final_sentence, synonyms_dict

# # Example usage
# if __name__ == "__main__":
#     sentence = input("Enter a sentence: ")
#     keywords, synonyms = process_text(sentence)
#     print("Processed Sentence:", keywords)
#     print("Synonyms:", synonyms)
