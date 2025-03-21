import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Step 1: Load and Preprocess the Dataset
def process_data(file_path, seq_length=50):
    # Load the text file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().lower()  

    # Create a unique character set
    unique_chars = sorted(list(set(content)))
    char_to_index = {char: index for index, char in enumerate(unique_chars)}
    index_to_char = {index: char for index, char in enumerate(unique_chars)}

    # Prepare sequences of characters and their corresponding outputs
    input_sequences, output_chars = [], []
    for i in range(0, len(content) - seq_length):
        sequence_in = content[i:i + seq_length]
        sequence_out = content[i + seq_length]
        input_sequences.append([char_to_index[char] for char in sequence_in])
        output_chars.append(char_to_index[sequence_out])

    # Reshape and normalize the input data
    input_sequences = np.reshape(input_sequences, (len(input_sequences), seq_length))
    output_chars = to_categorical(output_chars, num_classes=len(unique_chars))

    return input_sequences, output_chars, unique_chars, char_to_index, index_to_char

# Step 2: Build the LSTM Model
def create_lstm_model(input_shape, output_classes):
    model = Sequential([
        Embedding(input_dim=output_classes, output_dim=50, input_length=input_shape),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(256),
        Dropout(0.2),
        Dense(output_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Step 3: Train the Model
def train_lstm_model(model, X, y, epochs=50, batch_size=128):
    # Save the best model during training
    checkpoint = ModelCheckpoint("lstm_text_generator.keras", monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    return model

# Step 4: Generate Text
def generate_text_sequence(model, seed, char_to_index, index_to_char, seq_length, chars_to_generate=100, temp=1.0):
    generated_output = seed
    for _ in range(chars_to_generate):
        # Convert seed text to integer sequence
        sequence = [char_to_index[char] for char in seed]
        sequence = pad_sequences([sequence], maxlen=seq_length, truncating='pre')

        # Predict the next character
        predictions = model.predict(sequence, verbose=0)[0]
        predictions = np.log(predictions) / temp
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)
        next_char = index_to_char[np.argmax(np.random.multinomial(1, predictions, 1))]

        # Append the predicted character to the seed text
        generated_output += next_char
        seed = seed[1:] + next_char

    return generated_output

# Main Execution
if __name__ == "__main__":
    # Parameters
    file_path = "shakespeare.txt"  
    sequence_length = 100
    epochs = 50
    batch_size = 128

    # Step 1: Load and preprocess data
    X, y, unique_chars, char_to_index, index_to_char = process_data(file_path, sequence_length)

    # Step 2: Build the model
    model = create_lstm_model(X.shape[1], len(unique_chars))
    print(model.summary())

    # Step 3: Train the model
    model = train_lstm_model(model, X, y, epochs=epochs, batch_size=batch_size)

    model.save('lstm_text_generator.keras')
    model = tf.keras.models.load_model('lstm_text_generator.keras')

    # Step 4: Generate text
    seed_text = "shall i compare thee to a summer's day?\n"
    generated_text = generate_text_sequence(model, seed_text, char_to_index, index_to_char, sequence_length, chars_to_generate=500, temp=0.5)
    print("\nGenerated Text:\n", generated_text)
