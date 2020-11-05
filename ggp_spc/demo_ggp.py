import sys,argparse

import GGP.main_ggp as ggp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ggp', help='The model to use as a prior')

    parsed,sys.argv = parser.parse_known_args()
    sys.argv.insert(0,parsed.model)

    ggp.main()

