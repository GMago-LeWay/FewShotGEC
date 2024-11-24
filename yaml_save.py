import argparse
import yaml
import os

def main():
    parser = argparse.ArgumentParser(description='Save key-value pairs to a YAML file.')
    
    parser.add_argument('key_value_pairs', metavar='KEY=VALUE', type=str, nargs='+',
                        help='A key=value pair')
    
    parser.add_argument('--yaml', type=str, required=True,
                        help='Path to the YAML file where the data will be stored.')
    args = parser.parse_args()

    data = dict(item.split('=', 1) for item in args.key_value_pairs)

    # create result dir
    os.makedirs(os.path.dirname(args.yaml), exist_ok=True)

    with open(args.yaml, 'w') as file:
        yaml.dump(data, file)

if __name__ == '__main__':
    main()
