import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("../dataset/spam.csv", encoding="latin-1")

# Remove unwanted columns
data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

# Rename columns
data.columns = ["Label", "Message"]

# Convert labels into numbers
data["Label"] = data["Label"].map({"ham": 0, "spam": 1})

# Convert text into numbers using a stronger feature representation.
# This matches the prediction-time vectorizer expectations in app.py.
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2)
X = vectorizer.fit_transform(data["Message"])
y = data["Label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the model with a small smoothing parameter for improved
# class separation on spam keywords.
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
message = ["Hi, are you coming to college tomorrow?"]
# Convert message into TF-IDF
message_vector = vectorizer.transform(message)

# Predict
prediction = model.predict(message_vector)

# Display result
if prediction[0] == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: HAM") 