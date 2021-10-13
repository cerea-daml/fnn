
import numpy as np
from tensorflow.keras.models import load_model

def keras_to_txt(filename_in, filename_out, add_norm_in=False, norm_alpha_in=1, norm_beta_in=0, add_norm_out=False, norm_alpha_out=1, norm_beta_out=0):
    str_format = '{}'
    int_format = '{:>7d}'
    float_format = '{:0.7e}'
    model = load_model(filename_in)
    config = model.get_config()
    model_name = config['name']
    lines = []
    if 'sequential' in model_name:
        lines.append(str_format.format('sequential'))
        num_layers = len(model.layers)
        if add_norm_in:
            num_layers += 1
        if add_norm_out:
            num_layers += 1
        lines.append(int_format.format(num_layers))
        input_shape = config['layers'][0]['config']['batch_input_shape']
        if add_norm_in:
            lines.append(str_format.format('normalisation'))
            lines.append(int_format.format(input_shape[1]))
            lines.append(float_format.format(norm_alpha_in))
            lines.append(float_format.format(norm_beta_in))
        for (layer, subconfig) in zip(model.layers, config['layers'][1:]):
            if subconfig['class_name'] == 'Dense':
                lines.append(str_format.format('dense'))
                output_shape = layer.compute_output_shape(input_shape)
                lines.append(int_format.format(input_shape[1]))
                lines.append(int_format.format(output_shape[1]))
                kernel = layer.weights[0].numpy().flatten()
                bias = layer.weights[1].numpy()
                parameters = np.concatenate([bias, kernel])
                lines.append('\t'.join(float_format.format(num) for num in parameters))
                lines.append(str_format.format(subconfig['config']['activation']))
                input_shape = output_shape
            else:
                raise Exception(f'unsupported layer: {subconfig["class_name"]}')
        if add_norm_out:
            lines.append(str_format.format('normalisation'))
            lines.append(int_format.format(input_shape[1]))
            lines.append(float_format.format(norm_alpha_out))
            lines.append(float_format.format(norm_beta_out))
    else:
        raise Exception(f'unsupported model: {model_name}')

    with open(filename_out, 'w') as f:
        for line in lines:
            f.write(line+'\n')

