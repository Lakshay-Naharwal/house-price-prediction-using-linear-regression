import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def preprocess_sqft(x):
    tokens = str(x).split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

def train():
    # 1. Load Data
    df = pd.read_csv('data.csv')
    print(f"Initial shape: {df.shape}")

    # 2. Data Cleaning
    df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
    df3 = df2.dropna()

    # Size to BHK
    df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
    
    # Total Sqft to float
    df3['total_sqft'] = df3['total_sqft'].apply(preprocess_sqft)
    df3 = df3[df3.total_sqft.notnull()]

    # 3. Outlier Removal
    # Remove properties where sqft per bhk is less than 300
    df4 = df3[~(df3.total_sqft / df3.bhk < 300)]
    
    # Price per sqft
    df4['price_per_sqft'] = df4['price'] * 100000 / df4['total_sqft']
    
    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    df5 = remove_pps_outliers(df4)

    def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')

    df6 = remove_bhk_outliers(df5)
    
    # Remove bath outliers
    df7 = df6[df6.bath < df6.bhk + 2]
    
    # Drop price_per_sqft as it was used only for outlier detection
    df8 = df7.drop(['size', 'price_per_sqft'], axis='columns')

    # 4. Encoding Location
    df8.location = df8.location.apply(lambda x: x.strip())
    location_stats = df8.groupby('location')['location'].agg('count').sort_values(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df8.location = df8.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

    # Convert to Dummies
    dummies = pd.get_dummies(df8.location)
    df9 = pd.concat([df8, dummies.drop('other', axis='columns')], axis='columns')
    df10 = df9.drop('location', axis='columns')

    # 5. Split Data
    X = df10.drop(['price'], axis='columns')
    y = df10.price

    # Train-Val-Test Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=10)

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    # 6. Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 7. Metrics
    def get_metrics(X, y, name):
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        print(f"\n--- {name} Metrics ---")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        return r2, rmse, mae

    get_metrics(X_train, y_train, "Training")
    get_metrics(X_val, y_val, "Validation")
    get_metrics(X_test, y_test, "Test")

    # 8. Save Model and Metadata
    if not os.path.exists('model'):
        os.makedirs('model')
    
    joblib.dump(model, 'model/bangalore_house_price_model.pkl')
    
    # Save column info for prediction
    metadata = {
        'columns': X.columns.tolist()
    }
    joblib.dump(metadata, 'model/metadata.pkl')
    
    # Save locations for UI
    import json
    locations = sorted(df8.location.unique().tolist())
    with open('model/locations.json', 'w') as f:
        json.dump(locations, f)
        
    print("\nModel, metadata, and locations saved successfully")

if __name__ == "__main__":
    train()
