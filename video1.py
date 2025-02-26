import nltk
import spacy
import cv2
import os
import requests
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from PIL import Image
import io

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("stopwords")

# Initialize NLP components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

# Path to video assets
VIDEO_FOLDER = "assets"
DEFAULT_VIDEO = os.path.join(VIDEO_FOLDER, "hello.mp4")  # Default fallback video

# Bing Image Search API Key (Replace with your key)
BING_API_KEY = "YOUR_BING_API_KEY"
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/images/search"


def process_text(sentence):
    """Tokenize, lemmatize, and remove stopwords from the input sentence."""
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

        # Keep important words
        if word not in stop_words or word in ["is", "are", "was", "were", "be", "to", "i", "you", "we", "he", "she", "they", "will"]:
            processed_words.append(lemmatized_word)

    doc = nlp(" ".join(processed_words))
    final_sentence = " ".join([token.text for token in doc])

    return final_sentence.split()  # Return list of processed words


def fetch_image_for_word(word):
    """Fetch an image using Bing Image Search API."""
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": word, "count": 1, "imageType": "photo"}

    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    results = response.json()

    if "value" in results and len(results["value"]) > 0:
        return results["value"][0]["contentUrl"]
    return None


def play_videos_for_sentence(words):
    """Plays videos for words or fetches images if videos are missing."""
    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)

    for word in words:
        video_path = os.path.join(VIDEO_FOLDER, f"{word}.mp4")

        # If video exists, play it
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video file {video_path}")
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Stop when the video ends

                cv2.imshow("Video Player", frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):  # Press 'q' to exit early
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            cap.release()
        
        else:
            print(f"Video not found for '{word}', fetching an image instead.")

            # Fetch image from Bing Image Search
            image_url = fetch_image_for_word(word)
            if image_url:
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                img_np = np.array(image)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                cv2.imshow("Video Player", img_np)
                cv2.waitKey(2000)  # Show image for 2 seconds
            else:
                print(f"No image found for '{word}', skipping...")

    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    words = process_text(sentence)
    print("Processed Words:", words)

    play_videos_for_sentence(words)
