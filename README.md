# Sentiment Analysis of Amazon Reviews

This python script performs sentiment analysis on amazon reviews text to determine if a review is positive, neutral or negative. It utilises the following libraries:
spaCy - For advanced natural language processing tasks including tokenisation and lemmatisation.
pandas - for data manipulation and loading the data CSV file
TextBlob - For sentiment analysis using its pre-trained sentiment lexicons.

# Script Functionality:

Data Loading and Cleaning:
- Loads the product reviews from a CSV file using pandas.
- Drops rows with missing values in the "reviews.text" column.
- Normalises the review text using the normalise_text function:
- Removes non-alphanumeric characters (except underscores).
- Removes stop words, punctuation, and currency symbols.
- Removes words with length 2 or less (optional).
- Performs lemmatization (converting words to their base form) and converts text to lowercase.

Sentiment Prediction:
- Predicts sentiment for each review using the predict_sentiment function:
- Uses TextBlob to analyze the sentiment polarity of the normalized text.
- Classifies sentiment as positive, neutral, or negative based on the polarity score.

Analysis and Output:
- Prints the length of the predicted sentiment list and the selected review list (should be the same).
- Prints a sample of the selected reviews and their corresponding predicted sentiments.

## Usage

1. Clone the repository:
git clone https://github.com/TomD101/Coursework-projects.git

2. Install required libraries:
pip install -r requirements.txt

3. (Optional) Replace the amazon_product_reviews.csv file:
If you have your own CSV containing product reviews (ensure it has a "reviews.text" column), replace the existing file.

4. Run the script:
python sentiment_analysis_fianl.py # Replace with your actual script name if different

## Dependencies

* spacy (Version: 3.7.5)
* pandas (Version: 2.1.4)
* textblob (Version: 0.15.3)

## License (Optional)

No specific license applied
