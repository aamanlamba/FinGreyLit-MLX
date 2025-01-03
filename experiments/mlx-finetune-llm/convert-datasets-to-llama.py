import json
import os
from datetime import date

data_dir = './data'
print(os.getcwd())
print(data_dir)
for filename in os.listdir(data_dir):
    if filename.endswith('.jsonl'):
        input_filepath = os.path.join(data_dir, filename)
        output_filepath = os.path.join(data_dir, 'output_' + filename)

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
