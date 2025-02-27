import ollama
import requests
import cv2
import numpy as np
import time
from pydantic import BaseModel
import re

# Google API Credentials (Replace these with your credentials)
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID"
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

# Function to extract technical words using Llama 3.1
def extract_technical_terms(text):
    prompt = f"Extract all technical words from the following text:\n\n{text}\n\nOnly return the list of words."
    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    
    # Extract response content
    technical_words = response['message']['content'].split("\n")  

    # Clean extracted words (remove bullet points or unwanted characters)
    cleaned_words = [re.sub(r"[^\w\s]", "", word).strip() for word in technical_words if word.strip()]
    
    return cleaned_words

# Pydantic model for validating image search input
class ImageSearchInput(BaseModel):
    query: str

# Function to fetch an image using Google Custom Search API
def fetch_image(query: str) -> str:
    validated_input = ImageSearchInput(query=query)  # Validate input using Pydantic

    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": validated_input.query,
        "searchType": "image",
        "num": 1  # Get only one image
    }
    response = requests.get(GOOGLE_SEARCH_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        if "items" in data and data["items"]:
            return data["items"][0]["link"]  # Return the image URL
        else:
            print(f"No image found for {query}")
            return None
    else:
        print(f"Failed to fetch image for {query}. Status Code: {response.status_code}")
        return None

# Function to display images using OpenCV
def display_images(technical_words):
    for word in technical_words:
        print(f"Fetching image for: {word}")
        image_url = fetch_image(word)  # Fetch image URL
        time.sleep(1)  # Prevent API rate limiting
        
        if not image_url:
            print(f"No image found for {word}")
            continue

        # Fetch and display the image
        response = requests.get(image_url)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is not None:
                cv2.imshow(word, image)
                cv2.waitKey(2000)  # Show each image for 2 seconds
                cv2.destroyAllWindows()
            else:
                print(f"Failed to decode image for {word}")
        else:
            print(f"Failed to fetch image for {word}")

# Example usage
text = "The model utilizes transformers, attention mechanisms, and embeddings to process NLP tasks efficiently."
technical_words = extract_technical_terms(text)
print("Extracted Technical Words:", technical_words)
display_images(technical_words)
