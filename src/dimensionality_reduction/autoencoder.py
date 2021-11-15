from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


class AutoEncoder:

    def __init__(self,
                 encoder_dimensions=[250],
                 encoder_activations=["relu"],
                 decoder_dimensions=[512],
                 decoder_activations=["relu"],
                 full_model_loss="categorical_crossentropy",
                 full_model_optimizer=SGD(learning_rate=0.001),
                 full_model_metrics="accuracy",
                 compile=True):

        # predefining models
        self.encoder = Sequential(name="encoder")
        self.decoder = Sequential(name="decoder")

        # encoder parameters
        self.encoder_dimensions = encoder_dimensions
        self.encoder_activations = encoder_activations

        # decoder parameters
        self.decoder_dimensions = decoder_dimensions
        self.decoder_activations = decoder_activations

        # compilation parameters
        self.model_loss = full_model_loss
        self.model_optimizer = full_model_optimizer
        self.model_metrics = full_model_metrics

        # build encoder
        self.build_model(self.encoder, self.encoder_dimensions, self.encoder_activations)

        # build decoder
        self.build_model(self.decoder, self.decoder_dimensions, self.decoder_activations)

        self.full_model = Sequential([self.encoder, self.decoder])

        if compile is True:
            self.full_model.compile(loss=self.model_loss, optimizer=self.model_optimizer, metrics=self.model_metrics)

    def train(self, training_data, dev_data, epochs, batch_size=None):
        print('Fitting autoencoder...')
        history = self.full_model.fit(
            training_data, training_data, epochs=epochs, validation_data=(dev_data, dev_data), batch_size=batch_size)
        return history

    @staticmethod
    def build_model(model, dimensions, activations):
        for unit_n in dimensions:
            # if multiple activations are given, the class assumes there is one for each layer
            if len(activations) > 1:

                if len(activations) != len(dimensions):
                    raise ValueError("Provide an activation function for each autoencoder layer.")

                for activation in activations:
                    AutoEncoder.__add_layer(model, unit_n=unit_n, activation=activation)
            else:
                AutoEncoder.__add_layer(model, unit_n=unit_n, activation=activations[0])

    @staticmethod
    def __add_layer(model, unit_n, activation):
        layer = Dense(units=unit_n, activation=activation)
        model.add(layer)


def build_model():
    """ Compile autoencoder model. """

    encoder = Sequential(name="encoder")
    encoder.add(Dense(250, activation="relu", name="bottleneck"))
    # encoder.add(Dense(125, activation="relu"))
    # encoder.add(Dense(60, activation="relu"))
    # encoder.add(Dense(30, activation="relu", name="bottleneck"))

    decoder = Sequential(name="decoder")
    # decoder.add(Dense(60, input_shape=[30], activation='relu'))
    # decoder.add(Dense(125, activation='relu'))
    # decoder.add(Dense(250, activation='relu'))
    decoder.add(Dense(512, activation="sigmoid"))

    model = Sequential([encoder, decoder])
    model.compile(loss="binary_crossentropy",
                  optimizer=SGD(learning_rate=0.001),  # SGD 0.001 is pretty wholesome
                  metrics=['accuracy'])

    return model


def train_encoder(training_data, dev_data, n=10):
    """ Perform dimensionality reduction using autoencoder. """

    model = build_model()

    print("Fitting autoencoder.")
    model.fit(training_data, training_data, epochs=n, validation_data=(dev_data, dev_data))
    encoder = model.get_layer("encoder")
    print()

    return encoder
