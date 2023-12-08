import glob
import pandas as pd
import os
import numpy as np
import tgt # https://textgridtools.readthedocs.io/en/stable/api.html

def run(args):

    # Read TextGrid file
    tg_file = args.input_tg

    # Extract basename
    basename = os.path.basename(tg_file).replace('_checked.TextGrid', '')

    





def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--input_tg", type=str, help = "")
    parser.add_argument("--output_dir", type=str, help = "")
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()