import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess():
    data = pd.read_csv("dataset/marketing_campaign.csv")

    # Remove missing values
    data = data.dropna()

    # Convert categorical to numeric
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col])

    return data