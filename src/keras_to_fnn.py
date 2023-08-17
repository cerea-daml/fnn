"""! @brief Interface module between keras and FNN.

@details This package provides the \ref keras_file_to_txt and
\ref keras_to_txt functions to convert a keras model
into a txt/bin file which can be read by FNN.
The NN architecture is written in the txt file, the NN parameters
are written in the binary file.

Programmatic usage
------------------

Use the \ref keras_file_to_txt function as follows to convert a `.h5` file
into a `.txt` file:

    >>> keras_file_to_txt('model_out.txt', 'model_out.bin', 'model_in.h5')

Use the \ref keras_to_txt function as follows to convert a keras
model into a .txt/.bin file:

    >>> keras_to_txt('model_out.txt', 'model_out.bin', keras_model)

Command-line usage
------------------

Use the following command to convert a `.h5` file into a `.txt` file:

    >>> python keras_to_fnn.py model_out.txt model_out.bin model_in.h5
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
def get_model_name(model):
    """Returns the model name for a given model."""
    return model.get_config()['name']

def get_layer_name(subconfig):
    """Returns the layer name for a given layer."""
    return subconfig['class_name']

def get_num_layers(model, norm_in, norm_out):
    """Returns the total number of layers for a given model."""
    num_layers = 0
    config = model.get_config()
    for (layer, subconfig) in zip(model.layers, config['layers'][1:]):
        layer_name = get_layer_name(subconfig)
        if layer_name in ['Dense', 'Dropout']:
            num_layers += 1
    if norm_in is not None:
        num_layers += 1
    if norm_out is not None:
        num_layers += 1
    return num_layers

def get_input_shape(model):
    """Returns the input shape for a given model."""
    return model.get_config()['layers'][0]['config']['batch_input_shape']

def add_normalisation_layer(
    write_txt, 
    write_bin,
    input_shape, 
    freeze=True,
    alpha=None,
    beta=None,
):
    """Adds content for a normalisation layer."""
    name = 'frozen-normalisation' if freeze else 'normalisation'
    write_txt(STR_FORMAT(name))
    write_txt(INT_FORMAT(input_shape[1]))
    write_bin(alpha)
    write_bin(beta)

def add_layer(
    write_txt, 
    write_bin,
    input_shape, 
    layer, 
    subconfig,
):
    """Adds content for a layer."""
    layer_name = get_layer_name(subconfig)
    if layer_name == 'Dense':
        output_shape = layer.compute_output_shape(input_shape)
        kernel = layer.weights[0].numpy().flatten()
        bias = layer.weights[1].numpy()
        parameters = np.concatenate([bias, kernel])
        train = subconfig['config']['trainable']
        name = 'dense' if train else 'frozen-dense'
        activation = subconfig['config']['activation']
        write_txt(STR_FORMAT(name))
        write_txt(INT_FORMAT(input_shape[1]))
        write_txt(INT_FORMAT(output_shape[1]))
        write_txt(STR_FORMAT(activation))
        write_bin(parameters)
        return output_shape
    elif layer_name == 'Dropout':
        output_shape = layer.compute_output_shape(input_shape)
        dropout_rate = subconfig['config']['rate']
        write_txt(STR_FORMAT('dropout'))
        write_txt(INT_FORMAT(output_shape[1]))
        write_txt(FLOAT_FORMAT(dropout_rate))
        return output_shape
    raise UnsupportedLayerException(layer_name)
##\endcond

def keras_to_txt(
    filename_out_txt, 
    filename_out_bin, 
    model, 
    norm_in=None,
    norm_out=None,
):
    """! @brief Transforms a keras model into a txt/bin model file which can be read by FNN.

    @details The NN architecture is written in the txt file, the NN parameters
    are written in the binary file.
    @param[in] filename_out_txt The name of the txt file to write.
    @param[in] filename_out_bin The name of the bin file to write.
    @param[in] model The keras model.
    @param[in] norm_in Key-word arguments (freeze, alpha, beta) 
    for the optional input normalisation layer.
    @param[in] norm_out Key-word arguments (freeze, alpha, beta) 
    for the optional output normalisation layer.
    """
    with open(filename_out_txt, 'w') as file_out_txt:
        with open(filename_out_bin, 'wb') as file_out_bin:

            def write_txt(line):
                file_out_txt.write(line + '\n')

            def write_bin(p):
                p.astype('f').tofile(file_out_bin)

            model_name = get_model_name(model)

            if 'sequential' in model_name:

                num_layers = get_num_layers(model, norm_in, norm_out)
                input_shape = get_input_shape(model)

                write_txt(STR_FORMAT('sequential'))
                write_txt(INT_FORMAT(num_layers))

                if norm_in is not None:
                    add_normalisation_layer(
                        write_txt,
                        write_bin,
                        input_shape,
                        **norm_in,
                    )

                for (layer, subconfig) in zip(model.layers, model.get_config()['layers'][1:]):
                    input_shape = add_layer(
                        write_txt, 
                        write_bin,
                        input_shape, 
                        layer, 
                        subconfig,
                    )

                if norm_out is not None:
                    add_normalisation_layer(
                        write_txt,
                        write_bin,
                        input_shape,
                        **norm_out,
                    )
            else:
                raise UnsupportedModelException(model_name)


def keras_file_to_txt(
    filename_out_txt, 
    filename_out_bin, 
    filename_in, 
    custom_objects=None,
    **kwargs,
):
    """! @brief Transforms a keras model file into a txt/bin model file which can be read by FNN.

    @details This function is actually a wrapper around \ref keras_to_txt.
    It accepts the same kwargs, with the addition of `custom_objects`.


    \b Note

    This function uses `tf.keras.models.load_model` to read
    `filename_in`.
    @param[in] filename_out_txt The name of the txt file to write.
    @param[in] filename_out_bin The name of the bin file to write.
    @param[in] filename_in The keras model file.
    @param[in] custom_objects List of custom objects to ignore while reading keras file.
    @param[in] kwargs Key-word arguments forwarded to `keras_to_txt`.
    """
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    custom_objects = {key: None for key in custom_objects} if custom_objects else {}
    model = tf.keras.models.load_model(filename_in, custom_objects=custom_objects)
    keras_to_txt(filename_out_txt, filename_out_bin, model, **kwargs)

##\cond
if __name__ == '__main__':
    DESC = 'Transforms a keras model file into a txt/bin model file which can be read by FNN.'
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument('txt_file', help='output txt file (architecture)')
    parser.add_argument('bin_file', help='output bin file (parameters)')
    parser.add_argument('keras_file', help='input keras file')
    parser.add_argument('-c', '--custom', nargs='*', help='list custom keras object to ignore')
    args = parser.parse_args()
    keras_file_to_txt(
        args.txt_file, 
        args.bin_file, 
        args.keras_file,
        custom_objects=args.custom,
    )
##\endcond
