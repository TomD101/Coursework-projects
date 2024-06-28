import re # Regular expressions for handling symbols and irregular text
import spacy # Import the spacy library
import pandas as pd # Import pandas for data handling

# Load small word sentiment module
nlp = spacy.load('en_core_web_sm')

# Import dataframe
df = pd.read_csv('amazon_product_reviews.csv')

# Drop null values and select only the reviews.text column
amazon_df = df.dropna(subset=['reviews.text'])

# Data cleaning

# Load the reviews text column from dataset
reviews_data = amazon_df['reviews.text']
reviews_data.head(5) # view first few records to check they loaded correctly


def normalise_text(text):
  """
  This function normalises text for sentiment analysis.
  Args:
    text (str): The text data to be normalised.
  Returns:
    list: A list of normalised tokens.
  """

  # Expression for character filtering
  text = re.sub(r"[^A-Za-z0-9_]", " ", str(text))  # Allow letters, numbers, and underscores

  # Process text with spaCy
  doc = nlp(text)

  # List to store normalised tokens
  normalised_tokens = []

  for token in doc:
  # Filter based on text features to be removed
    if (token.is_stop  # Remove stop words
    or token.is_punct  # Remove punctuation
    or token.is_currency  # Remove currency symbols
    or len(token.text) <= 2):  # Remove words with length 2 or less
      continue


  # Lemmatisation and lowercase conversion
  normalised_tokens.append(token.lemma_.lower())

  return normalised_tokens # Output from function is list of lemmatised text

# Make list and call function for all reviews
normalised_reviews = []
for review in reviews_data:
  normalised_reviews.append(" ".join(normalise_text(review))) # Appends the lematised words back to strings before occupying list

# Import sentiment analysis module
from textblob import TextBlob

def predict_sentiment(text):
  """
  This function predicts sentiment of text.
  Args:
    text (str): The text data to have sentiment predicted.
  Returns:
    list: A list of sentiment tokens.
  """
  # Create a TextBlob object from the review text
  blob = TextBlob(text)
  # Get sentiment polarity
  sentiment = blob.sentiment.polarity

  # Classify sentiment based on polarity score
  if sentiment > 0:
    return "positive"
  elif sentiment < 0:
    return "negative"
  else:
    return "neutral"


# Example review to test sentiment analysis function
review = "The product didn't do what I expected it to" # Enter a review string to generate prediction of positive, neutral or negative
predicted_sentiment = predict_sentiment(review) # call function
print(f"Review: {review}") # print review
print(f"Predicted Sentiment: {predicted_sentiment}") # print sentiment

# Select specific reviews
selected_reviews = normalised_reviews[0:50]  # Select the first 50 reviews

# List to store predicted sentiments
predicted_sentiments = []

# Predict sentiment for each selected review
for review in selected_reviews:
  predicted_sentiment = predict_sentiment(review)
  predicted_sentiments.append(predicted_sentiment)

# Analyse predicted sentiments
print(f"Length of predicted_sentiments: {len(predicted_sentiments)}")
print(f"Length of DataFrame index: {len(selected_reviews)}")

# Print sample reviews and predicted sentiments
for i, (review, sentiment) in enumerate(zip(selected_reviews, predicted_sentiments)):
  print(f"Review {i+1}: {review} - Predicted Sentiment: {sentiment}") # Lists selected reviews followed by their sentiment label
