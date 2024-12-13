import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl
import os

# Ensure NLTK data directory exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Manual NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
nltk.download('stopwords', quiet=True, download_dir=nltk_data_dir)

class MovieDataProcessor:
    def __init__(self, data_dir='data'):
        """
        Initialize data processor with data directory
        """
        self.data_dir = data_dir
        self.movies_df = pd.read_csv(f'{data_dir}/movies.csv')
        self.ratings_df = pd.read_csv(f'{data_dir}/ratings.csv')
        self.links_df = pd.read_csv(f'{data_dir}/links.csv')
        self.tags_df = pd.read_csv(f'{data_dir}/tags.csv')
        
        self.integrated_df = None
    
    def preprocess_genres(self):
        """
        Preprocess movie genres
        """
        # Split genres
        self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
        
        # One-hot encode genres
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(self.movies_df['genres_list'])
        genre_columns = [f'genre_{genre}' for genre in mlb.classes_]
        genre_df = pd.DataFrame(genre_encoded, columns=genre_columns)
        
        self.movies_df = pd.concat([self.movies_df, genre_df], axis=1)
        return self
    
    def aggregate_ratings(self):
        """
        Aggregate rating information
        """
        ratings_agg = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count'],
            'userId': 'nunique'
        }).reset_index()
        ratings_agg.columns = ['movieId', 'avg_rating', 'total_ratings', 'unique_users']
        
        return ratings_agg
    
    def preprocess_tags(self):
        """
        Preprocess and aggregate tags
        """
        def preprocess_text(text):
            if pd.isna(text):
                return ''
            text = str(text).lower()  # Convert to string
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = text.split()  # Simple splitting instead of word_tokenize
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if w not in stop_words]
            return ' '.join(tokens)
        
        # Aggregate tags
        tags_agg = self.tags_df.groupby('movieId')['tag'].apply(
            lambda x: ' '.join(x.astype(str))  # Convert to string before joining
        ).reset_index()
        tags_agg.columns = ['movieId', 'combined_tags']
        tags_agg['processed_tags'] = tags_agg['combined_tags'].apply(preprocess_text)
        
        return tags_agg
    
    def handle_missing_values(self):
        """
        Handle missing values in the integrated dataset
        """
        # Print info about missing values
        print("Missing values before handling: ")
        print(self.integrated_df.isnull().sum())

        # Handle missing values in 'avg_rating'
        # Strategy: Remove rows with missing avg_rating
        self.integrated_df = self.integrated_df.dropna(subset=['avg_rating'])

        # For other columns, fill with median or a specific value
        numeric_columns = self.integrated_df.select_dtypes(include=[np.number]).columns
        self.integrated_df[numeric_columns] = self.integrated_df[numeric_columns].fillna(self.integrated_df[numeric_columns].median())

        # For categorical columns, fill with the most frequent value or 'Unknown'
        categorical_columns = self.integrated_df.select_dtypes(include=['object']).columns
        self.integrated_df[categorical_columns] = self.integrated_df[categorical_columns].fillna('Unknown')

        # Print info after handling missing values
        print("\nMissing values after handling: ")
        print(self.integrated_df.isnull().sum())

        return self.integrated_df
    
    def integrate_datasets(self):
        """
        Integrate all datasets
        """
        self.preprocess_genres()
        
        # Aggregate ratings and tags
        ratings_agg = self.aggregate_ratings()
        tags_agg = self.preprocess_tags()
        
        # Merge datasets
        integrated_df = self.movies_df.merge(
            ratings_agg, on='movieId', how='left'
        ).merge(
            tags_agg, on='movieId', how='left'
        ).merge(
            self.links_df, on='movieId', how='left'
        )
        
        self.integrated_df = integrated_df
        self.handle_missing_values()  # Add this line to handle missing values
        return self.integrated_df
    
    def extract_nlp_features(self):
        """
        Extract NLP features using TF-IDF
        """
        if self.integrated_df is None:
            raise ValueError("Integrate datasets first")
        
        # TF-IDF for titles
        title_vectorizer = TfidfVectorizer(max_features=50)
        title_features = title_vectorizer.fit_transform(
            self.integrated_df['title'].fillna('')
        )
        
        # TF-IDF for tags
        tags_vectorizer = TfidfVectorizer(max_features=50)
        tags_features = tags_vectorizer.fit_transform(
            self.integrated_df['processed_tags'].fillna('')
        )
        
        # Convert to DataFrames
        title_feature_df = pd.DataFrame(
            title_features.toarray(), 
            columns=[f'title_feature_{i}' for i in range(title_features.shape[1])]
        )
        tags_feature_df = pd.DataFrame(
            tags_features.toarray(), 
            columns=[f'tags_feature_{i}' for i in range(tags_features.shape[1])]
        )
        
        # Combine features
        self.integrated_df = pd.concat([self.integrated_df, title_feature_df, tags_feature_df], axis=1)
        
        return self.integrated_df
    
    def save_cleaned_dataset(self, output_file='cleaned_movie_data.csv'):
        """
        Save the cleaned and integrated dataset to a CSV file
        """
        if self.integrated_df is not None:
            self.integrated_df.to_csv(output_file, index=False)
            print(f"Cleaned dataset saved to {output_file}")
        else:
            print("No data to save.")
    
# Example usage
if __name__ == "__main__":
    processor = MovieDataProcessor()
    integrated_data = processor.integrate_datasets()  # Integrates datasets and handles missing values
    nlp_features = processor.extract_nlp_features()  # Extracts NLP features from titles and tags
    processor.save_cleaned_dataset(output_file='cleaned_movie_data.csv')  # Saves the final cleaned dataset to a file
