# Introduction
We want to construct a FNN (feedforward neural network) to predict spotify song popularity based on a few factors:
```'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'```
All of these are numerical values, so our data will be easy to work with. 
The target variable is ```'track_popularity'```

# Setup
We will first split the data into training and testing data using train test split. Then, we will scale the data so that our variables are equally contributing to the analysis.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

# Part 1: How does our model perform without any tuning?
We will construct a model with the ```relu``` and ```sigmoid``` activation function. We will use 50 epochs, batch size 32, and validation split 0.2 for consistency for our first trial.
```
model1 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1) 
])

model1.compile(optimizer='adam', loss='mean_squared_error')

model1.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

test_loss = model1.evaluate(X_test_scaled, y_test)

print('Test RMSE (relu):', (test_loss)**(0.5))

model2 = Sequential([
    Dense(64, activation='sigmoid', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='sigmoid'),
    Dense(1) 
])

model2.compile(optimizer='adam', loss='mean_squared_error')

model2.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

test_loss = model2.evaluate(X_test_scaled, y_test)
print('Test RMSE (sigmoid):', (test_loss)**(0.5))
```
$$\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2}
\]$$


Test RMSE (relu): 23.987563407778364
Test RMSE (sigmoid): 23.882521158674418

Our RMSE of ~24 is not bad, on average, our model is off by 24 in the prediction of track popularity, which ranges from 1-100. 
