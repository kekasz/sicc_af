import argparse
from find import f_invalid


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=str)
    return parser


def main():

    # GET PATH FROM CALLING ARGUMENT
    path = build_parser().parse_args().path

    # GET ERRORS FROM PATH FILE
    return f_invalid(path)


if __name__ == '__main__':
    main()
