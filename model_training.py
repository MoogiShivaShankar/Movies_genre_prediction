import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, classification_report

from dataintegration import MovieDataProcessor

class MovieModelTrainer:
    def __init__(self, integrated_df):
        self.integrated_df = integrated_df
    
    def prepare_features(self):
        feature_columns = [
            'avg_rating', 'total_ratings', 'unique_users',
            *[col for col in self.integrated_df.columns if col.startswith('genre_')],
            *[col for col in self.integrated_df.columns if col.startswith('title_feature_')],
            *[col for col in self.integrated_df.columns if col.startswith('tags_feature_')]
        ]
        
        X = self.integrated_df[feature_columns].fillna(0)
        return X
    
    def train_rating_model(self, X):
        y = self.integrated_df['avg_rating']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rating_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        rating_model.fit(X_train, y_train)
        
        y_pred = rating_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Rating Prediction MSE: {mse}")
        
        joblib.dump(rating_model, 'models/rating_model.pkl')
        
        return rating_model
    
    def train_genre_models(self, X):
        genre_columns = [col for col in self.integrated_df.columns if col.startswith('genre_')]
        
        genre_models = {}
        genre_results = {}
        
        for genre in genre_columns:
            y_genre = self.integrated_df[genre]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_genre, test_size=0.2, random_state=42)
            
            genre_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            genre_model.fit(X_train, y_train)
            
            y_pred = genre_model.predict(X_test)
            genre_results[genre] = classification_report(y_test, y_pred)
            genre_models[genre] = genre_model
        
        joblib.dump(genre_models, 'models/genre_models.pkl')
        
        return genre_models, genre_results

def main():
    # Create 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Load and process data
    processor = MovieDataProcessor()
    integrated_df = processor.integrate_datasets()
    integrated_df = processor.extract_nlp_features()
    
    # Train models
    trainer = MovieModelTrainer(integrated_df)
    X = trainer.prepare_features()
    
    # Train rating model
    rating_model = trainer.train_rating_model(X)
    
    # Train genre models
    genre_models, genre_results = trainer.train_genre_models(X)
    
    # Print genre classification results
    for genre, result in genre_results.items():
        print(f"\nGenre: {genre}")
        print(result)

if __name__ == "__main__":
    main()
