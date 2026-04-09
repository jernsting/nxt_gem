import networkx as nx
import numpy as np
from tensorflow.keras import backend as KBack
from tensorflow.keras.layers import Input, Lambda, Subtract
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

from gem.embedding.sdne_utils import (
    batch_generator_sdne,
    get_autoencoder,
    get_decoder,
    get_encoder,
    graphify,
    model_batch_predictor,
    savemodel,
    saveweights,
)
from gem.embedding.static_graph_embedding import StaticGraphEmbedding


class SDNE(StaticGraphEmbedding):
    hyper_params = {
        "method_name": "sdne",
        "actfn": "relu",
        "modelfile": None,
        "weightfile": None,
        "savefilesuffix": None,
    }

    def __init__(self, *args, **kwargs):
        """Initialize the SDNE class

        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            alpha: weighing hyperparameter for 1st order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            K: number of hidden layers in encoder/decoder
            n_units: vector of length K-1 containing #units in hidden layers
                     of encoder/decoder, not including the units in the
                     embedding layer
            rho: bounding ratio for number of units in consecutive layers (< 1)
            n_iter: number of sgd iterations for first embedding (const)
            xeta: sgd step size parameter
            n_batch: minibatch size for SGD
            modelfile: Files containing previous encoder and decoder models
            weightfile: Files containing previous encoder and decoder weights
        """
        super().__init__(*args, **kwargs)
        self.learned = False

    def learn_embedding(self, graph=None, is_weighted=False, no_python=False):
        if not graph:
            raise ValueError("graph needed")
        sparse = nx.to_scipy_sparse_array(graph)
        sparse = (sparse + sparse.T) / 2
        self._node_num = len(graph.nodes)

        # Generate encoder, decoder and autoencoder
        self._num_iter = self._n_iter
        # If cannot use previous step information, initialize new models
        self._encoder = get_encoder(
            self._node_num,
            self._d,
            self._K,
            self._n_units,
            self._nu1,
            self._nu2,
            self._actfn,
        )
        self._decoder = get_decoder(
            self._node_num,
            self._d,
            self._K,
            self._n_units,
            self._nu1,
            self._nu2,
            self._actfn,
        )
        self._autoencoder = get_autoencoder(self._encoder, self._decoder)

        # Initialize self._model
        # Input
        x_in = Input(shape=(2 * self._node_num,), name="x_in")
        x1 = Lambda(lambda x: x[:, 0 : self._node_num], output_shape=(self._node_num,))(
            x_in
        )
        x2 = Lambda(
            lambda x: x[:, self._node_num : 2 * self._node_num],
            output_shape=(self._node_num,),
        )(x_in)
        # Process inputs
        [x_hat1, y1] = self._autoencoder(x1)
        [x_hat2, y2] = self._autoencoder(x2)
        # Outputs
        x_diff1 = Subtract()([x_hat1, x1])
        x_diff2 = Subtract()([x_hat2, x2])
        y_diff = Subtract()([y2, y1])

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            """Hack: This fn doesn't accept additional arguments.
                  We use y_true to pass them.
            y_pred: Contains x_hat - x
            y_true: Contains [b, deg]
            """
            return (
                KBack.sum(KBack.square(y_pred * y_true[:, 0 : self._node_num]), axis=-1)
                / y_true[:, self._node_num]
            )

        def weighted_mse_y(y_true, y_pred):
            """Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
            y_pred: Contains y2 - y1
            y_true: Contains s12
            """
            min_batch_size = KBack.shape(y_true)[0]
            return (
                KBack.reshape(
                    KBack.sum(KBack.square(y_pred), axis=-1), [min_batch_size, 1]
                )
                * y_true
            )

        # Model
        self._model = Model(inputs=x_in, outputs=[x_diff1, x_diff2, y_diff])
        sgd = SGD(
            learning_rate=self._xeta, weight_decay=1e-5, momentum=0.99, nesterov=True
        )
        self._model.compile(
            optimizer=sgd,
            loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y],
            loss_weights=[1, 1, self._alpha],
        )

        feat_dim = sparse.shape[1]

        output_sig = (
            tf.TensorSpec(shape=(None, feat_dim * 2), dtype=tf.float32),  # InData
            (
                tf.TensorSpec(shape=(None, feat_dim + 1), dtype=tf.float32),  # a1
                tf.TensorSpec(shape=(None, feat_dim + 1), dtype=tf.float32),  # a2
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # X_ij
            )
        )

        ds = tf.data.Dataset.from_generator(
            lambda: batch_generator_sdne(sparse, self._beta, self._n_batch, True),
            output_signature=output_sig
        )

        self._model.fit(
            ds,
            epochs=self._num_iter,
            steps_per_epoch=sparse.nonzero()[0].shape[0] // self._n_batch,
            verbose=1,
        )
        # Get embedding for all points
        self._Y = model_batch_predictor(self._autoencoder, sparse, self._n_batch)
        # Save the autoencoder and its weights
        if self._weightfile is not None:
            saveweights(self._encoder, self._weightfile[0])
            saveweights(self._decoder, self._weightfile[1])
        if self._modelfile is not None:
            savemodel(self._encoder, self._modelfile[0])
            savemodel(self._decoder, self._modelfile[1])
        if self._savefilesuffix is not None:
            saveweights(
                self._encoder, "encoder_weights_" + self._savefilesuffix + ".weights.h5"
            )
            saveweights(
                self._decoder, "decoder_weights_" + self._savefilesuffix + ".weights.h5"
            )
            savemodel(self._encoder, "encoder_model_" + self._savefilesuffix + ".json")
            savemodel(self._decoder, "decoder_model_" + self._savefilesuffix + ".json")
            # Save the embedding
            np.savetxt("embedding_" + self._savefilesuffix + ".txt", self._Y)
        self.learned = True
        return self._Y

    def get_embedding(self, filesuffix=None):
        if not self.learned:
            raise ValueError("Embedding not learned yet")
        return (
            self._Y
            if filesuffix is None
            else np.loadtxt("embedding_" + filesuffix + ".txt")
        )

    def get_edge_weight(self, i, j, embed=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt("embedding_" + filesuffix + ".txt")
        if i == j:
            return 0
        else:
            sparse_hat = self.get_reconst_from_embed(embed[(i, j), :], filesuffix)
            return (sparse_hat[i, j] + sparse_hat[j, i]) / 2

    def get_reconstructed_adj(self, embed=None, node_l=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt("embedding_" + filesuffix + ".txt")
        sparse_hat = self.get_reconst_from_embed(embed, node_l, filesuffix)
        return graphify(sparse_hat)

    def get_reconst_from_embed(self, embed, node_l=None, filesuffix=None):
        if filesuffix is None:
            if node_l is not None:
                return self._decoder.predict(embed, batch_size=self._n_batch)[:, node_l]
            else:
                return self._decoder.predict(embed, batch_size=self._n_batch)
        else:
            try:
                decoder = model_from_json(
                    open("decoder_model_" + filesuffix + ".json").read()
                )
            except FileNotFoundError:
                print(
                    f"Error reading file: {'decoder_model_' + filesuffix + '.json'}. Cannot load previous model"
                )
                exit()
            try:
                decoder.load_weights("decoder_weights_" + filesuffix + ".weights.h5")
            except (FileNotFoundError, ReferenceError):
                print(
                    f"Error reading file: {'decoder_weights_' + filesuffix + '.weights.h5'}. Cannot load previous weights"
                )
                exit()
            if node_l is not None:
                return decoder.predict(embed, batch_size=self._n_batch)[:, node_l]
            else:
                return decoder.predict(embed, batch_size=self._n_batch)
