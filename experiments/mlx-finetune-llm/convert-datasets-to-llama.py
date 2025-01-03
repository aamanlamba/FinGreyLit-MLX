import json
import os
from datetime import date

# Get the directory of the current script 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Construct the full path to the target file 
target_file_path = os.path.join(script_dir, 'data')

print(target_file_path)
for filename in os.listdir(target_file_path):
    if filename.endswith('.jsonl') and filename.startswith('mlx'):
        input_filepath = os.path.join(target_file_path, filename)
        output_filepath = os.path.join(target_file_path, 'output_' + filename)

        with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
            for line in infile:
                entry = json.loads(line)
                convs = entry['conversations']
                output_text = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
                Cutting Knowledge Date: December 2023\n   \
                Today Date: {date.today()} \n\n  \
                You are a skilled librarian specialized in meticulous cataloguing of digital documents.\n   \
                <|eot_id|>\n <|start_header_id|>user<|end_header_id|>'
                
                for conv in convs:
                    from_type = conv['from']
                    value = conv['value']
                    
                    if from_type == 'system':
                        continue
                    elif from_type == 'user':
                        output_text += f'{value}\n    <|eot_id|>\n  \
                                <|start_header_id|>assistant<|end_header_id|>'
                    elif from_type == 'assistant':
                        output_text += f'{value}\n    '
                
                outfile.write(json.dumps({"text": output_text}) + '\n')

print("Conversion completed!")
