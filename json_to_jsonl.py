import json
import sys

def json_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    with open(output_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def convert_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.read()

    # Split the data by ';' to separate individual JSON objects
    objects = data.split('};')

    # Add back the closing brace '}' to each object except for the last one
    for i in range(len(objects) - 1):
        objects[i] += '}'

    with open(output_file, 'w') as f:
        # Write each object as a separate line in the output JSONL file
        for obj in objects:
            obj = obj.strip()
            if obj:  # Skip empty lines
                json.dump(json.loads(obj), f)  # Load and dump each object to ensure proper formatting
                f.write('\n')  # Add a newline character to separate each object

# Example usage:
# python json_to_jsonl ./HumanEval/HumanEvalComm.json ./HumanEval/HumanEvalComm.jsonl
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_jsonl.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_to_jsonl(input_file, output_file)
