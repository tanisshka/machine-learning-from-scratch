import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist  

(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
print(y_train)

x_train=x_train.reshape(-1,28*28).astype("float32")/255.0
x_test=x_test.reshape(-1,28*28).astype("float32")/255.0

#? Sequential API : The Sequential model, which is very straightforward (a simple list of layers), but is limited to single-input, single-output stacks of layers (as the name gives away).

model=keras.Sequential(
    [
        layers.Dense(512,activation='relu'),
        layers.Dense(256,activation='relu'),
        layers.Dense(10)
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)
model.evaluate(x_test,y_test,batch_size=32,verbose=2)


#? Functional API: The Functional API, which is an easy-to-use, fully-featured API that supports arbitrary model architectures. For most people and most use cases, this is what you should be using. This is the Keras "industry strength" model.

# Input
inputs = keras.Input(shape=(784,))

# Hidden layers
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(256, activation='relu')(x)

# Output with softmax
outputs = layers.Dense(10, activation='softmax')(x)

# Model
model1 = keras.Model(inputs=inputs, outputs=outputs)

# Compile (IMPORTANT CHANGE HERE)
model1.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Train
model1.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)

# Evaluate
model1.evaluate(x_test, y_test, batch_size=32, verbose=2)