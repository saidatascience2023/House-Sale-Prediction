# -*- coding: utf-8 -*-
"""Deploy.ipynb

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

st.title("🏠 House Price Predictor App")
st.write("1. Upload your training data (must include target: `SalePrice`)\n"
         "2. Then upload test data without `SalePrice`\n"
         "3. Get predictions!")

# Step 1: Upload training data
train_file = st.file_uploader("Upload Training CSV (with 'SalePrice')", type=["csv"], key="train")

if train_file is not None:
    train_df = pd.read_csv(train_file)
    st.subheader("Training Data Preview")
    st.write(train_df.head())

    if 'SalePrice' not in train_df.columns:
        st.error("Training data must contain a 'SalePrice' column.")
    else:
        # Separate features and target
        X_train = train_df.drop(['SalePrice', 'Id'], axis=1, errors='ignore')
        y_train = train_df['SalePrice']

        # Preprocessing for numerical & categorical features
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        # Final pipeline with Random Forest
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        model.fit(X_train, y_train)

        st.success("✅ Model trained successfully on uploaded training data!")

        # Step 2: Upload test data
        test_file = st.file_uploader("Upload Test CSV (without 'SalePrice')", type=["csv"], key="test")

        if test_file is not None:
            test_df = pd.read_csv(test_file)
            st.subheader("Test Data Preview")
            st.write(test_df.head())

            X_test = test_df.drop(['Id'], axis=1, errors='ignore')

            try:
                predictions = model.predict(X_test)
                result_df = test_df.copy()
                result_df['PredictedPrice'] = predictions

                st.subheader("📈 Predicted Prices")
                st.write(result_df[['Id', 'PredictedPrice']] if 'Id' in result_df else result_df)

                csv = result_df.to_csv(index=False)
                st.download_button("Download Predictions as CSV", csv, "predicted_prices.csv", "text/csv")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
