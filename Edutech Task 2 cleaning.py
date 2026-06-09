import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/datasets/Titanic-Dataset.csv")

# 1. Check missing values
print(df.isnull().sum())

# 2. Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# 3. Remove duplicates
df.drop_duplicates(inplace=True)

# 4. Normalize numeric columns
df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
df['Fare'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())

# 5. Handle outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

# Save cleaned dataset
df.to_csv('titanic_cleaned.csv', index=False)
print("Done! Cleaned file saved.")