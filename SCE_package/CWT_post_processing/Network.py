import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys


class MultiHeadSelfAttention(tf.keras.layers.Layer):

    """This class implements the human attention mechanism, which is also implemeted as a standard block in tensorflow framework.
    """

    def __init__(self, embed_dim, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This states the number of kind of features that are going to be used.
        self.num_heads = num_heads

        # This states the embed dimension of the features.
        self.embed_dim = embed_dim

        # embed_dim must be divisible by number of heads.
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads

        # query_dense for extracting features used as query
        self.query_dense = keras.layers.Dense(
            embed_dim, name='query', kernel_regularizer='l2')

        # key_dense for extracting features used as key
        self.key_dense = keras.layers.Dense(
            embed_dim, name='key', kernel_regularizer='l2')

        # value_dense for extracting features used as value
        self.value_dense = keras.layers.Dense(
            embed_dim, name='value', kernel_regularizer='l2')

        # combine_heads for combining the feature heads
        self.combine_heads = keras.layers.Dense(
            embed_dim, name='out', kernel_regularizer='l2')

    def attention(self, query, key, value):
        """The attention block implements the attention mechanism  using query, key and value features.

        Args:
            query : Query features extracted from the data.
            key : Key features extracted from the data.
            value : Value features extracted from the data.

        Returns:
            output, weights : The values after being processed by the attention mechanism, and its corresponding attention weights
        """
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        """This function serves to separate the features into multiple heads.

        Args:
            x : The features to be separated into multiple heads.
            batch_size : Batch size used in the learning task.

        Returns:
            The features separated into multiple heads.
        """
        x = tf.reshape(
            x, (batch_size, 112, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        """This function runs the above instructions.

        Args:
            inputs : Input features to be processed.

        Returns:
            output : Output after being processed by this attention mechanism.
        """
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, 112, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

    def get_config(self):
        """Standard block to get configurations from.

        Returns:
            dict : A dictionary containing the configurations of the block.
        """
        return {'num_heads': self.num_heads, 'embed_dim': self.embed_dim}

    @classmethod
    def from_config(cls, config):
        """Standard block to reconstruct the class from the configurations.

        Args:
            config : A dictionary for the configuration

        Returns:
            cls: A constructed class using the configurations.
        """
        return cls(**config)


class TransformerBlock(tf.keras.layers.Layer):
    """A standard transformer block that uses the attention mechanism.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.2):
        super(TransformerBlock, self).__init__()

        # Specify the dropout rate.
        self.dropout = dropout

        # Construct the attention mechanism.
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)

        # construct a simple feedforward network.
        self.ffn = tf.keras.Sequential(
            [keras.layers.Dense(ff_dim, activation='relu'),
             keras.layers.Dense(embed_dim)]
        )

        # Introduce normalization and dropout as regularization techniques.
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        """Standard block to get configurations from.

        Returns:
            dict : A dictionary containing the configurations of the block.
        """
        return {'embed_dim': self.embed_dim, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim}

    @classmethod
    def from_config(cls, config):
        """Standard block to reconstruct the class from the configurations.

        Args:
            config : A dictionary for the configuration

        Returns:
            cls: A constructed class using the configurations.
        """
        return cls(**config)


if __name__ == '__main__':

    # Get the data
    # Add the package to the python execution path
    sys.path.append(
        r'C:\Users\ASUS\Desktop\RWTH 1semester\RWTH 4th semester\SCE\SCE_package')

    import load_wavelet_scaleograms
    import PCA
    import data_normalize
    import train_test_splits

    # Load data and pred labels.
    data, pred = load_wavelet_scaleograms.load_data_and_pred()

    # Crop part of the data
    data, start_index, end_index = PCA.data_crop(data)

    # Use PCA to transform the data.
    data, pred = PCA.PCA_fit_transform(data, pred, start_index, end_index)

    # Normalize the data
    data = data_normalize.normalize(data)

    # Split the train and test data from the original dataset.
    data, pred, validation_data, validation_pred = train_test_splits.train_test_splits(
        data, pred)

    # Define the model

    inputs = keras.layers.Input(shape=(112, 100))

    x = keras.layers.Dense(units=32, activation='relu',
                           kernel_regularizer='l2')(inputs)

    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.LayerNormalization(axis=-1)(x)

    x = TransformerBlock(embed_dim=8, num_heads=1, ff_dim=8)(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(units=8, activation='relu',
                           kernel_regularizer='l2')(x)

    x = keras.layers.LayerNormalization(axis=-1)(x)

    x = keras.layers.Dense(units=2, activation='relu',
                           kernel_regularizer='l2')(x)

    model = keras.Model(inputs=inputs, outputs=x)

    print(model.summary())

    model.compile(
        # Compile the model with optimizers, loss types and metrics for monitoring purposes.
        optimizer=keras.optimizers.Adam(learning_rate=0.001),

        loss=keras.losses.MeanSquaredError(),

        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    # Use callback to prevent overfitting
    callback = keras.callbacks.EarlyStopping(
        monitor='val_mean_absolute_error', verbose=1, patience=200, mode='min', restore_best_weights=True)

    history = model.fit(
        data,
        pred,
        batch_size=150,
        epochs=1000,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(validation_data, validation_pred), shuffle=True, callbacks=[callback]
    )

    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
