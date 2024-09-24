import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server 
from copy import deepcopy

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')
known_locations = set()

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        for review in reviews:
            self.add_sentiment(review)
            known_locations.add(review["Location"])
        
        self.sort_reviews()

    def sort_reviews(self):
        reviews.sort(key=lambda review: review["sentiment"]["compound"] if review else print(review), reverse=True)

    def add_sentiment(self, review: dict):
        review["sentiment"] = self.analyze_sentiment(review.get("ReviewBody"))

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores
    
    def handle_get_request(self, environ: dict[str, Any]) -> dict:
        response_body = deepcopy(reviews)

        params = environ["QUERY_STRING"]
        params = parse_qs(params)

        requested_location = params.get("location")
        start_date = params.get("start_date")
        end_date = params.get("end_date")

        if requested_location:
            response_body = list(filter(lambda review: review["Location"] in requested_location, response_body))
        
        if start_date:
            start_date = datetime.strptime(start_date[-1], "%Y-%m-%d")
            response_body = list(filter(lambda review: datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") >= start_date, response_body))
        
        if end_date:
            end_date = datetime.strptime(end_date[-1], "%Y-%m-%d")
            response_body = list(filter(lambda review: datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") <= end_date, response_body))
        
        return response_body

    def handle_post_request(self, environ: dict[str, Any]) -> dict:
        size = int(environ["CONTENT_LENGTH"])
        params = environ["wsgi.input"].read(size).decode('utf-8')
        params = parse_qs(params)

        location = params.get("Location")
        review_body = params.get("ReviewBody")

        if location is None:
            raise ValueError("Location parameter is not optional")
        location = location[-1]
        
        if location not in known_locations:
            raise ValueError("Invalid Location parameter")

        if review_body is None:
            raise ValueError("ReviewBody parameter is not optional")
        review_body = review_body[-1]

        # print(location, review_body)
        review_id = uuid.uuid4()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "ReviewId": str(review_id),
            "ReviewBody": review_body,
            "Location": location,
            "Timestamp": timestamp
        }


    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Write your code here
            response_body = self.handle_get_request(environ)

            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(response_body, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                review = self.handle_post_request(environ)
                response_body = json.dumps(review, indent=2).encode("utf-8")
                
                self.add_sentiment(review)
                reviews.append(review)
                self.sort_reviews()
                start_response("201 OK", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
            except Exception as e:
                start_response("400 Missing property", [
                    ("Content-Type", "text/plain"),
                ])
                return [str(e).encode("utf-8")]
        

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()