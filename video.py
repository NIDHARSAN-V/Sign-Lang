import nltk
import spacy
import cv2
import os
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

# Path to the asset folder containing videos
VIDEO_FOLDER = "assets"
DEFAULT_VIDEO = os.path.join(VIDEO_FOLDER, "hello.mp4")  # Default video


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

    return final_sentence.split()  # Return as a list of words


def play_videos_for_sentence(words):
    """Plays videos for each word in a single OpenCV window."""
    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)

    for word in words:
        video_path = os.path.join(VIDEO_FOLDER, f"{word}.mp4")

        # Check if video exists, otherwise use default
        if not os.path.exists(video_path):
            print(f"Video not found for '{word}', playing default video.")
            video_path = DEFAULT_VIDEO

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

    # Keep the window open for 2 seconds after last video
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    words = process_text(sentence)
    print("Processed Words:", words)

    play_videos_for_sentence(words)
