import sys
import yaml

def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
            for key, value in data.items():
                yield f"export {key}='{value}'"
    except FileNotFoundError:
        print(f"Error: The file {yaml_file} does not exist.", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error in YAML file: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_yaml.py <path-to-yaml-file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    
    # output environments setting command
    for command in load_yaml(yaml_file):
        print(command)
