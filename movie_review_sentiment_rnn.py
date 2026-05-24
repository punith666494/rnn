# -----------------------------------
# 1. Load Dataset
# -----------------------------------

from tensorflow.keras.datasets import imdb

vocab_size = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))


# -----------------------------------
# 2. Padding Sequences
# -----------------------------------

from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 200

x_train = pad_sequences(
    x_train,
    maxlen=maxlen
)

x_test = pad_sequences(
    x_test,
    maxlen=maxlen
)

print(x_train.shape)


# -----------------------------------
# 3. Create Model
# -----------------------------------

from tensorflow.keras.models import Sequential

model = Sequential()


# -----------------------------------
# 4. Add Embedding Layer
# -----------------------------------

from tensorflow.keras.layers import Embedding

model.add(
    Embedding(
        input_dim=10000,
        output_dim=128,
        input_shape=(200,)
    )
)


# -----------------------------------
# 5. Add LSTM Layer
# -----------------------------------

from tensorflow.keras.layers import LSTM

model.add(
    LSTM(64)
)


# -----------------------------------
# 6. Add Output Layer
# -----------------------------------

from tensorflow.keras.layers import Dense

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)


# -----------------------------------
# 7. Compile Model
# -----------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# -----------------------------------
# 8. View Model Summary
# -----------------------------------

model.summary()


# -----------------------------------
# 9. Train Model
# -----------------------------------

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)
loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print("Loss:", loss)
print("Accuracy:", accuracy)
