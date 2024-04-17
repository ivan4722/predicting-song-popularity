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
$\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2}
\]$


Test RMSE (relu): 23.987563407778364\
Test RMSE (sigmoid): 23.882521158674418

Our RMSE of ~24 is not bad, on average, our model is off by 24 in the prediction of track popularity, which ranges from 1-100. 

#Part 2: Model Tuning
Since relu and sigmoid perform relatively similar, we can just choose 1 of the 2 activation functions for simplicity for our tuning. Lets use relu. 
We can define a function so we can easily tune the model in a loop.

```
ret = []
def build_model(epochs, learning_rate, batch_size):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1) 
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    test_loss = model.evaluate(X_test_scaled, y_test)
    test_rmse = (test_loss)**0.5
    s=f"Epochs: {epochs} Learning rate: {learning_rate} batch_size: {learning_rate}"
    ret.append(s)

```

We will first try adjusting the epoch. We can try epoch from 25 to 75 like so:
```
for epoch in range(25, 76, 5):
    build_model(epoch, 0.001, 32)
```
Epochs: 25 Learning rate: 0.001 batch_size: 0.001 rmse: 23.939526982554135
Epochs: 30 Learning rate: 0.001 batch_size: 0.001 rmse: 23.953614231794436
Epochs: 35 Learning rate: 0.001 batch_size: 0.001 rmse: 23.931665411646552
Epochs: 40 Learning rate: 0.001 batch_size: 0.001 rmse: 24.02880529461918
Epochs: 45 Learning rate: 0.001 batch_size: 0.001 rmse: 24.02321773273462
Epochs: 50 Learning rate: 0.001 batch_size: 0.001 rmse: 23.93933066596422
Epochs: 55 Learning rate: 0.001 batch_size: 0.001 rmse: 23.931892395563132
Epochs: 60 Learning rate: 0.001 batch_size: 0.001 rmse: 24.019353057254946
Epochs: 65 Learning rate: 0.001 batch_size: 0.001 rmse: 24.081379565898523
Epochs: 70 Learning rate: 0.001 batch_size: 0.001 rmse: 24.09851201319285
Epochs: 75 Learning rate: 0.001 batch_size: 0.001 rmse: 24.10152071511775
We can see there is not much improvement with increasing epoch. 

Lets adjust the learning_rate next.

```
lr = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1]
for rate in lr:
    build_model(25, rate, 32)
```


Epochs: 25 Learning rate: 1e-07 batch_size: 1e-07 rmse: 48.7292273451596
Epochs: 25 Learning rate: 1e-06 batch_size: 1e-06 rmse: 47.57277012148074
Epochs: 25 Learning rate: 1e-05 batch_size: 1e-05 rmse: 25.660343861108185
Epochs: 25 Learning rate: 0.0001 batch_size: 0.0001 rmse: 24.09503178334955
Epochs: 25 Learning rate: 0.001 batch_size: 0.001 rmse: 23.95020978672473
Epochs: 25 Learning rate: 0.01 batch_size: 0.01 rmse: 23.9834257751903
Epochs: 25 Learning rate: 0.1 batch_size: 0.1 rmse: 24.21431789082585

It seems simply changing the parameters does not result in large improvements in RMSE.  

WIP

