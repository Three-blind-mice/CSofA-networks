import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-f', '--file',
        dest='model_file',
        metavar='F',
        default='None',
        help='The load model from file')
    args = argparser.parse_args()
    return args