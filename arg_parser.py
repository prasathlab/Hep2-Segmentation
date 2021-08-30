import argparse

def get_parser():
    train_parser = argparse.ArgumentParser(allow_abbrev=False, description='Get inputs for Classification Pipeline')
    # Add the arguments
    train_parser.add_argument('--yaml',
                              dest='yaml_file',
                              required=True,
                              type=str,
                              help='Path to yaml file'
                              )
    train_parser.add_argument('--base_res_dir',
                              dest='base_res_dir',
                              required=True,
                              type=str,
                              help='Base Directory Path'
                              )

    train_parser.add_argument('--train_model',
                              dest='train_model',
                              action='store_true',
                              default=False,
                              help='If true: Train a Model else only Test model.'
                                   'If false, then test_res_dir and model_weights must be provided'
                              )

    train_parser.add_argument('--test_dir',
                              dest='test_dir',
                              required=False,
                              type=str,
                              help='Directory to save Test Results')



    train_parser.add_argument('--model_weights',
                              dest='model_weights',
                              required=False,
                              type=str,
                              help='Path to the model weights file.')



    return train_parser

