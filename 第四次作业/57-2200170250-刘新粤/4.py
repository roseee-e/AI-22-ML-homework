from sklearn.naive_bayes import GaussianNB
import numpy as np
outlook = np.array(['sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain', 'overcast', 'sunny', 'sunny', 'rain', 'sunny', 'overcast', 'overcast', 'rain'])
temperature = np.array(['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'])
humidity = np.array(['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'])
wind = np.array(['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong'])
play_tennis = np.array(['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'])
outlook_encoded = np.array([0 if x == 'sunny' else 1 if x == 'overcast' else 2 for x in outlook])
temperature_encoded = np.array([0 if x == 'hot' else 1 if x == 'mild' else 2 for x in temperature])
humidity_encoded = np.array([0 if x == 'high' else 1 for x in humidity])
wind_encoded = np.array([0 if x == 'weak' else 1 for x in wind])
play_tennis_encoded = np.array([0 if x == 'no' else 1 for x in play_tennis])
train_data = np.column_stack((outlook_encoded, temperature_encoded, humidity_encoded, wind_encoded))
train_target = play_tennis_encoded
model = GaussianNB()
model.fit(train_data, train_target)
test_data = np.array([0, 1, 0, 1])
prediction = model.predict([test_data])
if prediction == 0:
    print("Do not play tennis.")
else:
    print("Play tennis.")