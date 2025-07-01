import tensorflow as tf
import numpy as np

# Expanded medicine database (add your own entries)
MEDICINES = [
    "paracetamol", "dolo", "amoxicillin", "ibuprofen", 
    "aspirin", "cetirizine", "omeprazole", "atorvastatin"
]

# Character-level tokenizer
VOCAB = {char: idx+1 for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789- ")}
VOCAB_SIZE = len(VOCAB) + 1
MAX_LEN = 20

def vectorize_text(text):
    """Convert text to numerical vector"""
    text = text.lower().strip()
    vec = [VOCAB.get(c, 0) for c in text if c in VOCAB]
    return vec[:MAX_LEN] + [0] * (MAX_LEN - len(vec))

# Build and train model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 16, input_length=MAX_LEN),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(MEDICINES), activation='softmax')
])

# Prepare training data
X_train = np.array([vectorize_text(name) for name in MEDICINES])
y_train = tf.keras.utils.to_categorical(range(len(MEDICINES)))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, verbose=0)

def validate_medicine(text, threshold=0.3):
    """Check if text resembles known medicine"""
    vec = np.array([vectorize_text(text)])
    pred = model.predict(vec, verbose=0)[0]
    max_prob = np.max(pred)
    return max_prob > threshold, MEDICINES[np.argmax(pred)], float(max_prob)
