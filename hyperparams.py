from tensorflow.keras import optimizers

batch_size = 32
epochs = 10
activation="sigmoid"
optimizer=optimizers.Adam(1e-4)
loss='binary_crossentropy'
metrics=["accuracy"]

#how many clusters?
n_clusters = 4
#how many folds for cross-fold?
num_folds = 4