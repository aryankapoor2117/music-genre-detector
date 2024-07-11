import json
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
import math

DATASET_PATH = "Data.json"
MODEL_PATH = "trained_model.h5"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    #convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def create_model(input_shape):
    model = keras.Sequential([
        #input layer
        keras.layers.Flatten(input_shape=input_shape),

        # 1st hidden layer
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    return model

def extract_mfcc(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    DURATION = 30
    SAMPLES_PER_SEGMENT = sr * DURATION

    # Process segments extracting mfcc
    num_samples_per_segment = int(SAMPLES_PER_SEGMENT / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    mfcc_segments = []

    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment

        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_mfcc=n_mfcc,
                                    n_fft=n_fft,
                                    hop_length=hop_length)
        mfcc = mfcc.T

        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            mfcc_segments.append(mfcc.tolist())

    return np.array(mfcc_segments)

def get_user_input():
    while True:
        file_path = input("\nEnter the path to your audio file (or 'q' to quit): ").strip()
        if file_path.lower() == 'q':
            return 'q'
        if os.path.exists(file_path) and file_path.lower().endswith(('.mp3', '.wav', '.m4a')):
            return file_path
        else:
            print("Invalid file path or unsupported audio format. Please try again.")

def analyze_audio(file_path, model, genres):
    SAMPLE_RATE = 22050
    DURATION = 30
    SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION

    # Load audio file
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Split the audio into 30-second segments
    num_segments = math.ceil(len(signal) / SAMPLES_PER_SEGMENT)
    segments = [signal[i*SAMPLES_PER_SEGMENT:(i+1)*SAMPLES_PER_SEGMENT] for i in range(num_segments)]

    results = []

    for i, segment in enumerate(segments):
        # Pad or truncate the segment to exactly 30 seconds
        if len(segment) < SAMPLES_PER_SEGMENT:
            segment = np.pad(segment, (0, SAMPLES_PER_SEGMENT - len(segment)))
        elif len(segment) > SAMPLES_PER_SEGMENT:
            segment = segment[:SAMPLES_PER_SEGMENT]

        # Extract MFCCs
        mfcc = extract_mfcc(segment, SAMPLE_RATE)

        if mfcc.shape[1:] != (130, 13):
            print(f"Warning: MFCC shape mismatch in segment {i+1}. Skipping this segment.")
            continue

        # Classify the segment
        predictions = model.predict(mfcc)
        predicted_index = np.argmax(np.mean(predictions, axis=0))
        confidence = np.mean(predictions, axis=0)[predicted_index]

        results.append({
            'segment': i+1,
            'genre': genres[predicted_index],
            'confidence': confidence
        })

    return results

if __name__ == "__main__":

    #load data
    inputs, targets = load_data(DATASET_PATH)

    #split data
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.1)

# check if model exists
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = keras.models.load_model(MODEL_PATH)
    else:
        print("Creating and training new model...")
        model = create_model((inputs.shape[1], inputs.shape[2]))

        #compile network
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    
    
        model.summary()

        #train network

        history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test),
              epochs=100,
              batch_size=32)
    
        # plot accuracy and error over the epochs
        plot_history(history)

        # save the model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=2)
    print(f"\nTest accuracy: {test_accuracy:.2f}")

    # Define genres (adjust this list if your genres are different)
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    while True:
        print("\nLet's classify your audio file!")
        print("Enter 'q' to quit the program.")
        
        test_file_path = get_user_input()
        
        if test_file_path.lower() == 'q':
            print("Thank you for using the music genre classifier. Goodbye!")
            break

        # Analyze the audio file
        results = analyze_audio(test_file_path, model, genres)

        # Sort results by confidence (high e xst first)
        results.sort(key=lambda x: x['confidence'], reverse=True)

        print("\nPredictions for all segments:")
        print("-" * 50)
        print("Segment  Genre       Confidence")
        print("-" * 50)

        for result in results:
            print(f"{result['segment']}        {result['genre']}    {result['confidence']:.2f}")

        print("-" * 50)
        print("\nBest prediction (highest confidence):")
        print(f"Segment: {results[0]['segment']}")
        print(f"Predicted genre: {results[0]['genre']}")
        print(f"Confidence: {results[0]['confidence']:.2f}")

        # Calculate and display the most common genre
        genre_counts = {}
        for result in results:
            genre_counts[result['genre']] = genre_counts.get(result['genre'], 0) + 1
        most_common_genre = max(genre_counts, key=genre_counts.get)

        print(f"\nMost common genre across all segments: {most_common_genre}")

        print("\n" + "="*50 + "\n")
