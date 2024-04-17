import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('spotify_songs.csv')


X = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
y = data['track_popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ret = []

def build_model(epochs, learning_rate, batch_size, hidden_layers):
    model = Sequential()

    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(X_train_scaled.shape[1],)))

    for layer_size in hidden_layers[1:]:
        model.add(Dense(layer_size, activation='relu'))
    
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    test_loss = model.evaluate(X_test_scaled, y_test)
    test_rmse = (test_loss)**0.5
    
    s = f"Epochs: {epochs}, Learning rate: {learning_rate}, Batch size: {batch_size}, Hidden layers: {hidden_layers}, RMSE: {test_rmse}"
    ret.append(s)
    return s

hidden_layers_config = [64] 
print(build_model(epochs=50, learning_rate=0.001, batch_size=32, hidden_layers=hidden_layers_config))