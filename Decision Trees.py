import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import time




df = pd.read_csv("GlobalWeatherRepository.csv")
relevant_columns = ['temperature_celsius', 'condition_text', 'wind_kph', 'wind_direction',
                    'pressure_mb', 'precip_mm', 'humidity', 'cloud', 'feels_like_celsius']
df = df[relevant_columns]
df.dropna(inplace=True)

label_encoders = {}

for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

label_encoder = LabelEncoder()

df['condition_text_encoded'] = label_encoder.fit_transform(df['condition_text'])
df.drop('condition_text', axis=1, inplace=True)
X = df.drop('condition_text_encoded', axis=1) 
y = df['condition_text_encoded']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)


classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)


temperature = float(input("Enter temperature in Celsius: "))
wind_kph = float(input("Enter wind speed in km/h: "))
wind_direction = input("Enter wind direction (e.g., NW, NE, SW, SE): ")
pressure_mb = float(input("Enter atmospheric pressure in mb: "))
precip_mm = float(input("Enter precipitation in mm: "))
humidity = float(input("Enter humidity percentage: "))
cloud = float(input("Enter cloud cover percentage: "))
feels_like_celsius = float(input("Enter feels-like temperature in Celsius: "))


wind_direction_encoded = label_encoders['wind_direction'].transform([wind_direction])[0]


new_data = pd.DataFrame({
    'temperature_celsius': [temperature],
    'wind_kph': [wind_kph],
    'wind_direction': [wind_direction],
    'pressure_mb': [pressure_mb],
    'precip_mm': [precip_mm],
    'humidity': [humidity],
    'cloud': [cloud],
    'feels_like_celsius': [feels_like_celsius]
})

for column in new_data.columns:
    if new_data[column].dtype == 'object':
        new_data[column] = label_encoders[column].transform(new_data[column])

predicted_condition_encoded = classifier.predict(new_data)

predicted_condition = label_encoders['condition_text'].inverse_transform(predicted_condition_encoded)

print("The weather condition probably will be:", predicted_condition[0])

print("Classification Report:")
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



feature_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()



train_sizes, train_scores, test_scores = learning_curve(classifier, X, y, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.show()