import os
import torch
import re


def read_styles(domain, output_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'styles')):
    style_fp = os.path.join(output_directory, f'{domain}-styles.pt')
    style_mean_fp = os.path.join(output_directory, f'{domain}-styles-mean.pt')
    style_std_fp = os.path.join(output_directory, f'{domain}-styles-std.pt')

    styles = torch.load(style_fp, map_location='cpu')
    return styles


def read_styles_mean_std(domain, output_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                               'styles')):
    style_fp = os.path.join(output_directory, f'{domain}-styles.pt')
    style_mean_fp = os.path.join(output_directory, f'{domain}-styles-mean.pt')
    style_std_fp = os.path.join(output_directory, f'{domain}-styles-std.pt')

    means = torch.load(style_mean_fp, map_location='cpu')
    stds = torch.load(style_std_fp, map_location='cpu')

    return means, stds


def read_clip_image_features(domain, output_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   'styles')):
    feature_path = os.path.join(output_directory, f'{domain}-clipfeatures.pt')
    features = torch.load(feature_path, map_location='cpu')
    return features


def style_layer_names(domain, model=None):
    layer_names = []
    # if model is not None:
    #     layer_names = [k for k, _ in model.named_parameters() if re.match("synthesis\.+(.*)+\.affine\.weight", k)]
    #     layer_names = [[s.split('.')[1]] for s in layer_names]
    #     print(layer_names)
    #     return layer_names

    if domain == 's3_landscape256':
        layer_names = [
        ['L0_36_512'],
        ['L1_36_512'],
        ['L2_36_512'],
        ['L3_52_512'],
        ['L4_52_512'],
        ['L5_84_512'],
        ['L6_84_512'],
        ['L7_148_512'],
        ['L8_148_512'],
        ['L9_148_362'],
        ['L10_276_256'],
        ['L11_276_181'],
        ['L12_276_128'],
        ['L13_256_128'],
        ['L14_256_3']
         ]
    else:
        layer_names = [
            ['b4', 'conv1'],
            ['b8', 'conv0'],
            ['b8', 'conv1'],
            ['b16', 'conv0'],
            ['b16', 'conv1'],
            ['b32', 'conv0'],
            ['b32', 'conv1'],
            ['b64', 'conv0'],
            ['b64', 'conv1'],
            ['b128', 'conv0'],
            ['b128', 'conv1'],
            ['b256', 'conv0'],
            ['b256', 'conv1'],
            ['b512', 'conv0'],
            ['b512', 'conv1'],
            ['b1024', 'conv0'],
            ['b1024', 'conv1'],
        ]

        if domain.endswith("256"):
            layer_names = layer_names[:-4]
        elif domain.endswith("512"):
            layer_names = layer_names[:-2]

    return layer_names


if __name__ == '__main__':
    means, stds = read_styles_mean_std('ffhq')

    for k, std in stds.items():
        print(std.shape)
