from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(data):
    X = data.drop("Response", axis=1)
    y = data["Response"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test