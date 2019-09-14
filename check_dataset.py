import argparse
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MSRA', 
        help="choose from MSRA, ICVL, NYU, HAND17"    
    )
    args = parser.parse_args()

    Dataset = getattr(datasets, "{}Dataset".format(args.dataset))
    dataset = Dataset()

    print('Data ready!')