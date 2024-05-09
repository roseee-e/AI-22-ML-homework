import pandas as pd

# Creating the dataset
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

# Creating DataFrame from dataset
df = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

# Function to calculate the prior probability of a class
def prior_probability(df, class_value):
    total = len(df)
    class_count = df['PlayTennis'].value_counts().get(class_value, 0)
    return class_count / total

# Function to calculate the conditional probability
def likelihood(df, feature_name, feature_value, class_value):
    class_df = df[df['PlayTennis'] == class_value]
    total_class = len(class_df)
    feature_count = class_df[feature_name].value_counts().get(feature_value, 0)
    return feature_count / total_class

# Function to predict class based on features
def predict_class(df, observation):
    # Get unique class labels
    class_labels = df['PlayTennis'].unique()
    # Dictionary to store probability results for each class
    probabilities = {}
    # Calculate probabilities for each class
    for label in class_labels:
        prob = prior_probability(df, label)
        for feature, value in observation.items():
            prob *= likelihood(df, feature, value, label)
        probabilities[label] = prob
    # Return the class with the highest probability
    return max(probabilities, key=probabilities.get)

# Define new observation and classify
new_observation = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

# Classify the new observation
result = predict_class(df, new_observation)
print("The prediction result is:", result)