from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_model = LogisticRegression()
    rf_model = RandomForestClassifier()

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    return log_model, rf_model, X_test, y_test
