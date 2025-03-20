import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout

# Load dataset
file_path = "Parkinsson disease M.csv"  # Update with the correct path if needed
df = pd.read_csv(file_path)

# Drop 'name' column
df = df.drop(columns=['name'])

# Split features and target
X = df.drop(columns=['status'])
y = df['status']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# -------------------- Train Random Forest --------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# -------------------- Prepare Data for LSTM and RNN --------------------
X_train_rnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# -------------------- Define LSTM Model --------------------
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile and train LSTM model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_rnn, y_train, epochs=50, batch_size=16, validation_data=(X_test_rnn, y_test), verbose=1)
lstm_preds = (lstm_model.predict(X_test_rnn) > 0.5).astype("int32").flatten()

# -------------------- Define RNN Model --------------------
rnn_model = Sequential([
    SimpleRNN(64, return_sequences=True, input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    SimpleRNN(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile and train RNN model
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train_rnn, y_train, epochs=50, batch_size=16, validation_data=(X_test_rnn, y_test), verbose=1)
rnn_preds = (rnn_model.predict(X_test_rnn) > 0.5).astype("int32").flatten()

# -------------------- Ensemble Model (Majority Voting) --------------------
ensemble_preds = np.round((rf_preds + lstm_preds + rnn_preds) / 3).astype(int)

# -------------------- Model Evaluation --------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))

# Print results
evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("LSTM", y_test, lstm_preds)
evaluate_model("RNN", y_test, rnn_preds)
evaluate_model("Ensemble (Random Forest + LSTM + RNN)", y_test, ensemble_preds)
