import numpy as np
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from Evaluation import evaluation


# Function to compute SPWVD
def compute_spwvd(signal):
    # Compute SPWVD using signal processing techniques
    # This could involve segmenting the signal, computing WVD, and smoothing

    # Placeholder for SPWVD computation
    spwvd = np.random.rand(10, 10)  # Replace with actual computation

    return spwvd


def Model_PDCNNet(Train_Data, Train_Target, Test_Data, Test_Target):
    # Compute SPWVD for each signal
    spwvd_data = np.array([compute_spwvd(signal) for signal in Train_Data])

    # Define CNN architecture
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=Train_Data.shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Train_Data.reshape(len(Train_Data), -1)).reshape(Train_Data.shape)
    X_test_scaled = scaler.transform(Test_Data.reshape(len(Test_Data), -1)).reshape(Test_Data.shape)

    # Train the model
    model.fit(X_train_scaled, Train_Target, epochs=10, batch_size=32, validation_split=0.1)

    pred = model.predict(X_test_scaled)
    Eval = evaluation(pred, Test_Target)
    return pred, Eval
