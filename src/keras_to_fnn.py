"""! @brief Interface module between keras and FNN.

@details This package provides the \ref keras_file_to_txt and
\ref keras_to_txt functions to convert a keras model
into a txt file which can be read by FNN.

Programmatic usage
------------------

Use the \ref keras_file_to_txt function as follows to convert a `.h5` file
into a `.txt` file:

    >>> keras_file_to_txt('model_out.txt', 'model_in.h5')

Use the \ref keras_to_txt function as follows to convert a keras
model into a .txt file:

    >>> keras_to_txt('model_out.txt', keras_model)

Command-line usage
------------------

Use the following command to convert a `.h5` file into a `.txt` file:

    >>> python keras_to_fnn.py model_out.txt model_in.h5
"""

import argparse
import numpy as np

## Format function for strings.
STR_FORMAT = '{}'.format
## Format function for integers.
INT_FORMAT = '{:>7d}'.format
## Format function for real numbers.
FLOAT_FORMAT = '{:0.7e}'.format


class UnsupportedModelException(Exception):
    """! Exception class for unsupported models."""


class UnsupportedLayerException(Exception):
    """! Exception class for unsupported layers."""


##\cond
def get_model_name(_model, config, **_kwargs):
    """Returns the model name for a given model."""
    return config['name']


def get_num_layers(model, _config, **kwargs):
    """Returns the total number of layers for a given model."""
    num_layers = len(model.layers)
    key = 'add_norm_in'
    if key in kwargs and kwargs[key]:
        num_layers += 1
    key = 'add_norm_out'
    if key in kwargs and kwargs[key]:
        num_layers += 1
    for rate in kwargs['dropout_rates']:
        if rate is not None:
            num_layers += 1
    return num_layers


def get_input_shape(_model, config, **_kwargs):
    """Returns the input shape for a given model."""
    return config['layers'][0]['config']['batch_input_shape']


def get_dropout_rates(model, _config, **kwargs):
    """Returns the dropout rates for each layer."""
    num_layers = len(model.layers)
    rates = kwargs.get('dropout_rates', None)
    if isinstance(rates, int):
        return [rates for _ in range(num_layers)]
    if isinstance(rates, list):
        if len(rates) > num_layers:
            raise ValueError('too many dropout rates are provided')
        while len(rates) < num_layers:
            rates.append(None)
        return rates
    return [None for _ in range(num_layers)]


def get_layer_name(_layer, subconfig, **_kwargs):
    """Returns the layer name for a given layer."""
    return subconfig['class_name']


def get_activation_name(_layer, subconfig, **_kwargs):
    """Returns the activation name for a given layer."""
    return subconfig['config']['activation']


def add_normalisation_layer(write, input_shape, for_input, **kwargs):
    """Adds content for a normalisation layer."""
    lbl = 'in' if for_input else 'out'
    key = f'add_norm_{lbl}'
    if key in kwargs and kwargs[key]:
        alpha = kwargs.get(f'norm_alpha_{lbl}', 1)
        beta = kwargs.get(f'norm_beta_{lbl}', 0)
        write(STR_FORMAT('normalisation'))
        write(INT_FORMAT(input_shape[1]))
        write('\t'.join(FLOAT_FORMAT(a) for a in alpha))
        write('\t'.join(FLOAT_FORMAT(b) for b in beta))


def add_layer(write, input_shape, layer, subconfig, dropout_rate, **kwargs):
    """Adds content for a layer."""
    layer_name = get_layer_name(layer, subconfig, **kwargs)
    if layer_name == 'Dense':
        output_shape = layer.compute_output_shape(input_shape)
        kernel = layer.weights[0].numpy().flatten()
        bias = layer.weights[1].numpy()
        parameters = np.concatenate([bias, kernel])
        write(STR_FORMAT('dense'))
        write(INT_FORMAT(input_shape[1]))
        write(INT_FORMAT(output_shape[1]))
        write('\t'.join(FLOAT_FORMAT(num) for num in parameters))
        write(STR_FORMAT(get_activation_name(layer, subconfig, **kwargs)))
        if dropout_rate is not None:
            write(STR_FORMAT('dropout'))
            write(INT_FORMAT(output_shape[1]))
            write(FLOAT_FORMAT(dropout_rate))
        return output_shape
    raise UnsupportedLayerException(layer_name)
##\endcond


def keras_to_txt(filename_out, model, **kwargs):
    """! @brief Transforms a keras model into a txt model file which can be read by FNN.

    @details \b Accepted \b kwargs
    - [in] `add_norm_in` : bool
        - Whether to add a normalisation layer for the input.
    - [in] `norm_alpha_in` : np.ndarray
        - Value of `alpha` (1d array) for the input normalisation layer, if any.
    - [in] `norm_beta_in` : np.ndarray
        - Value of `beta` (1d array) for the input normalisation layer, if any.
    - [in] `add_norm_out` : bool
        - Whether to add a normalisation layer for the output.
    - [in] `norm_alpha_out` : np.ndarray
        - Value of `alpha` (1d array) for the output normalisation layer, if any.
    - [in] `norm_beta_out` : np.ndarray
        - Value of `beta` (1d array) for the output normalisation layer, if any.
    - [in] `dropout_rates` : None or float or list of float
        - Whether to add dropout after each internal layer, with the given rate.

    @param[in] filename_out The name of the txt file to write.
    @param[in] model The keras model.
    @param[in] kwargs Key-word arguments.
    """
    with open(filename_out, 'w') as file_out:

        def write(line):
            file_out.write(line + '\n')

        config = model.get_config()
        model_name = get_model_name(model, config, **kwargs)
        if 'sequential' in model_name:
            input_shape = get_input_shape(model, config, **kwargs)
            dropout_rates = get_dropout_rates(model, config, **kwargs)
            kwargs['dropout_rates'] = dropout_rates
            num_layers = get_num_layers(model, config, **kwargs)
            write(STR_FORMAT('sequential'))
            write(INT_FORMAT(num_layers))
            add_normalisation_layer(write, input_shape, True, **kwargs)
            for (layer, dropout_rate, subconfig) in zip(model.layers, 
                    dropout_rates, config['layers'][1:]):
                input_shape = add_layer(write, input_shape, layer, subconfig,
                                        dropout_rate, **kwargs)
            add_normalisation_layer(write, input_shape, False, **kwargs)
        else:
            raise UnsupportedModelException(model_name)


def keras_file_to_txt(filename_out, filename_in, **kwargs):
    """! @brief Transforms a keras model file into a txt model file which can be read by FNN.

    @details This function is actually a wrapper around \ref keras_to_txt.
    It accepts the same kwargs.

    \b Note

    This function uses `tf.keras.models.load_model` to read
    `filename_in`.
    @param[in] filename_out The name of the txt file to write.
    @param[in] filename_in The keras model file.
    @param[in] kwargs Key-word arguments.
    """
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    model = tf.keras.models.load_model(filename_in)
    keras_to_txt(filename_out, model, **kwargs)


##\cond
if __name__ == '__main__':
    DESC = 'Transforms a keras model file into a txt model file which can be read by FNN.'
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument('txt_file', help='output txt file')
    parser.add_argument('keras_file', help='input keras file')
    args = parser.parse_args()
    keras_file_to_txt(args.txt_file, args.keras_file)
##\endcond
