import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
import time

warnings.filterwarnings('ignore')

df = pd.read_csv("GlobalWeatherRepository.csv")

relevant_columns = ['temperature_celsius', 'wind_kph', 'wind_direction',
                    'pressure_mb', 'precip_mm', 'humidity', 'cloud', 'feels_like_celsius', 'condition_text']

df = df[relevant_columns]

df.dropna(inplace=True)

label_encoder = LabelEncoder()
df['wind_direction_encoded'] = label_encoder.fit_transform(df['wind_direction'])

X = df.drop(['condition_text', 'wind_direction'], axis=1)
y = df['condition_text']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
start_time = time.time()

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
training_time = time.time() - start_time
print("Training Time: {:.2f} seconds".format(training_time))

print("Classification Report:")
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

def get_test_report(model):
    return classification_report(y_test, y_pred)

def plot_confusion_matrix(model):
    cm = confusion_matrix(y_test, y_pred)

    conf_matrix = pd.DataFrame(data=cm,
                               columns=df['condition_text'].unique(),
                               index=df['condition_text'].unique())
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=ListedColormap(['lightskyblue']),
                cbar=False, linewidths=0.1, annot_kws={'size': 25})

    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    plt.xlabel('Predicted Weather', fontsize=20, fontweight='bold')
    plt.ylabel('Actual Weather', fontsize=20, fontweight='bold')

    plt.show()

temperature = float(input("Enter temperature in Celsius: "))
wind_kph = float(input("Enter wind speed in km/h: "))
wind_direction = input("Enter wind direction (e.g., NW, NE, SW, SE): ")
pressure_mb = float(input("Enter atmospheric pressure in mb: "))
precip_mm = float(input("Enter precipitation in mm: "))
humidity = float(input("Enter humidity percentage: "))
cloud = float(input("Enter cloud cover percentage: "))
feels_like_celsius = float(input("Enter feels-like temperature in Celsius: "))

user_input = pd.DataFrame({
    'temperature_celsius': [temperature],
    'wind_kph': [wind_kph],
    'pressure_mb': [pressure_mb],
    'precip_mm': [precip_mm],
    'humidity': [humidity],
    'cloud': [cloud],
    'feels_like_celsius': [feels_like_celsius]
})

user_input['wind_direction_encoded'] = label_encoder.transform([wind_direction])[0]

scaled_user_input = scaler.transform(user_input)

predicted_weather = classifier.predict(scaled_user_input)

print("Predicted weather condition:", predicted_weather[0])


def plot_feature_importance(model):
    feature_importances = pd.Series(model.coef_[0], index=X.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

plot_feature_importance(classifier)
