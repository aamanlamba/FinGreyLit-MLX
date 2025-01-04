# %% [markdown]
# # Fine-tune meta-llama/Llama-3.2-1B-Instruct using mlx framework and llm-datasets

# %% [markdown]
# 

# %%
'''
import subprocess
import sys
import os

def install_requirements():
    # Get the path to the Python executable and use it to find pip
    python_executable = sys.executable
    pip_executable = os.path.join(os.path.dirname(python_executable), 'pip')
    subprocess.check_call([pip_executable, 'install', '-r', 'requirements.txt'])

if __name__ == '__main__':
    install_requirements()
'''
# %%
# set model name etc.

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT_NAME = MODEL_NAME.split('/')[-1]
SUFFIX = "FinGreyLit-MLX"
#SLICE = 1
print(MODEL_NAME)
print(MODEL_SHORT_NAME)

# %%
# Load and prepare fine-tuning dataset

import json
import glob
import random
import os

random.seed(42)  # for deterministic sampling of test set
# Get the directory of the current script 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Construct the full paths to the train and test files
trainfiles_path = os.path.join(script_dir, '../../llm-dataset/*-train.jsonl') 
testfiles_path = os.path.join(script_dir, '../../llm-dataset/*-test.jsonl')
train_files = glob.glob(trainfiles_path) 
test_files = glob.glob(testfiles_path)
print(train_files)
print(test_files)

EVAL_SIZE = 6  # how many documents to evaluate (i.e. calculate loss) on during fine-tuning
SYSTEM_PROMPT = "You are a skilled librarian specialized in meticulous cataloguing of digital documents."
INSTRUCTION = "Extract metadata from this document. Return as JSON."

def preprocess_sample(sample):
    output = json.dumps(sample["ground_truth"])
    input_ = json.dumps(sample["content"])
    # ShareGPT format
    conversations = [
        {'from': 'system', 'value': SYSTEM_PROMPT},
        {'from': 'user', 'value': INSTRUCTION + "\n\n" + input_},
        {'from': 'gpt', 'value': output}
    ]
    return {"conversations": conversations}

def dataset_to_records(files):
    records = []
    for filename in files:
        # Use glob to expand any patterns in the filenames 
        for file in glob.glob(filename):
            with open(filename) as infile:
                for line in infile:
                    sample = json.loads(line)
                    records.append(preprocess_sample(sample))
    return records

def write_jsonl(records, filename):
    with open(filename, "w") as outfile:
        for record in records:
            json.dump(record, outfile)
            outfile.write("\n")

train_recs = dataset_to_records(train_files)
random.shuffle(train_recs)
print(len(train_recs))
# Get the directory of the current script 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Construct the full path to the target file 
target_file_path = os.path.join(script_dir, 'data', 'mlx-train.jsonl') 
# Ensure the directory exists 
os.makedirs(os.path.join(script_dir, 'data'), exist_ok=True)
print(target_file_path)
print(len(train_recs))
write_jsonl(train_recs, target_file_path)
print(f"Wrote {len(train_recs)} train records")
target_file_path = os.path.join(script_dir, 'data', 'mlx-test.jsonl') 
print(target_file_path)

test_recs = dataset_to_records(test_files)
print(len(test_recs))
write_jsonl(test_recs, target_file_path)
print(f"Wrote {len(test_recs)} test records")
target_file_path = os.path.join(script_dir, 'data', 'mlx-eval.jsonl') 
print(target_file_path)
if EVAL_SIZE <= len(test_recs):
    eval_recs = random.sample(test_recs, EVAL_SIZE)
    print(len(eval_recs))
    write_jsonl(eval_recs, target_file_path)
    print(f"Wrote {len(eval_recs)} eval records")

# %%
import subprocess
# Get the directory of the current script 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Construct the full path to the target file 
target_file_path = os.path.join(script_dir, "convert-datasets-to-llama.py") 
subprocess.run(["python", target_file_path])

# %%
# Load and finetune LLM
from mlx_lm import generate,load
from huggingface_hub import login

import os, sys
from dotenv import load_dotenv
load_dotenv()

# Initialize the client with your API key
try:
    hf_accesstoken = os.environ.get("mlx_finetuning_api_key")
    login(hf_accesstoken)
except Exception as e:
    print(str(e))
    sys.exit(1)

# %%
#load model
model, tokenizer = load("meta-llama/Llama-3.2-1B-Instruct") 
#load("google/gemma-2-2b-it")
#generate prompt and response
prompt = "You are a skilled librarian \
        specialized in meticulous cataloguing of digital documents.\
        Extract metadata from this document. Return as JSON.\
        \{\"pdfinfo\": {\"author\": \"Johanna Elisabet Glader\", \
        \"creationDate\": \"D:20200226153952+02'00'\", \"modDate\": \
        \"D:20200226154128+02'00'\"}, \"pages\": [{\"page\": 1, \"text\": \
        \"This is an electronic reprint of the original article. \
        This reprint may differ from the original in pagination and typographic \
        detail.\\nPlease cite the original version:\\nCynthia S\\u00f6derbacka (2019). \
            Energy storage demo environment in Technobothnia. Vaasa\\nInsider, 11.12.2019.\"}, \
                {\"page\": 2, \"text\": \"Energy Storage Demo Environment in\\nTechnobothnia\\nWhen \
                discourses about the benefits that come with renewable energy sources come up, \
                    one cannot avoid the discussion going into the irregular tendencies of these \
                        sources especially with reference to solar and wind energy and the challenges \
                            that come with that.\\nUnfortunately, the future does not have much room for \
                                fossil fuels, the era of renewable energy sources is here and will continue to grow. \
                                    According to the International Renewable\\nNovia University of Applied Sciences \
                                        (Novia UAS), \\u00c5bo Akademi University (\\u00c5A), and\\nVaasa University \
                                            of Applied Sciences (VAMK) have partnered up under \
                                                the \\u201cEnergy\\nEducation & Research Laboratory.\"},\
                                                      {\"page\": 4, \"text\": \"A thesis worker from \\u00c5A is \
                                                      focusing on delivering a Phase Changing Materials \
                                                        (PCM)\"}, {\"page\": 5, \"text\": \"The Lab Engineer at \
                                                            Novia UAS is in charge of this.\\nAuthor: \
                                                                Cynthia S\\u00f6derbacka- Project Manager \
                                                                    & Lecturer at Novia UAS\"}]}\"}, \
                                                                        {\"from\": \"gpt\", \"value\": \"{\"language\": \
                                                                        \"en\", \"title\": \"Energy storage demo environment in \
                                                                        Technobothnia\", \"creator\": [\"S\\u00f6derbacka, Cynthia\"],\
                                                                              \"year\": \"2019\", \"publisher\": [\"Vaasa Insider\"],\
                                                                                \"type_coar\": \"newspaper article\"}\"}]}    "
print(prompt)
messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens = 100)
print(response)

# %%
# Load datasets
from datasets import load_dataset
# Get the directory of the current script 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Construct the full path to the target file 
traintarget_file_path = os.path.join(script_dir, 'data', 'output_mlx-train.jsonl') 
testtarget_file_path = os.path.join(script_dir, 'data', 'output_mlx-test.jsonl') 
evaltarget_file_path = os.path.join(script_dir, 'data', 'output_mlx-eval.jsonl') 
# Ensure the directory exists 
os.makedirs(os.path.join(script_dir, 'data'), exist_ok=True)
print(traintarget_file_path)
ds = load_dataset("json", 
                  data_files={"train": traintarget_file_path, 
                              "test": testtarget_file_path, 
                              "eval": evaltarget_file_path})

# convert dataset to dataframe
import pandas as pd
df_train = pd.DataFrame(ds['train'])
df_test = pd.DataFrame(ds['test'])
df_eval = pd.DataFrame(ds['eval'])
print(df_train.head())
print(df_test.head())
print(df_eval.head())
# convert dataframe to list for mlx

def preprocess(dataset):
    return dataset["text"].tolist()

train_set = preprocess(df_train)
dev_set = preprocess(df_eval)
test_set = preprocess(df_test)
#df_train['text'].tolist()
#dev_set = df_eval['text'].tolist()
#test_set = df_test['text'].tolist()

#train_list = df_train.to_dict(orient='records')
#test_list = df_test.to_dict(orient='records')
#eval_list = df_eval.to_dict(orient='records')
#train_set, dev_set, test_set = map(preprocess, (df_train,df_eval,df_test))

print(train_set[0])
print(dev_set[0])
print(test_set[0])

# %%
# Model finetuning setup
import matplotlib.pyplot as plt
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.tuner import train, TrainingArgs 
from mlx_lm.tuner import linear_to_lora_layers
from pathlib import Path
import json, time

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent
print(script_dir)

# Construct the full path to the target file
adapter_path = script_dir / 'adapters'
print(adapter_path)

# Ensure the directory exists
# adapter_path.mkdir(parents=True, exist_ok=True)

adapter_file_path = adapter_path / 'adapter_config.json'
print(adapter_file_path)

#set LORA parameters
lora_config = {
 "lora_layers": 8,
 "num_layers": 8,
 "lora_parameters": {
    "rank": 8,
    "scale": 20.0,
    "dropout": 0.0,
}}
# Check if adapter_config.json exists
if adapter_file_path.exists():
    print(f"The file '{adapter_file_path}' already exists.")
else:
    with open(adapter_file_path, "x") as fid:
        json.dump(lora_config, fid, indent=4)
        print(f"Created '{adapter_file_path}'")

# Set training parameters
training_args = TrainingArgs(
    adapter_file=adapter_path / "adapters.safetensors",
    iters=10, # number of iterations
    steps_per_eval=50 # number of steps per evaluation
)
#Freeze base model
model.freeze()

linear_to_lora_layers(model, lora_config["lora_layers"], lora_config["lora_parameters"])
num_train_params = (
    sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
)
print(f"Number of trainable parameters: {num_train_params}")

#Free up memory
import gc
gc.collect()
import pandas as pd
pd.reset_option('all')


# %%
# setup training model and optimizer - Adam
model.train()
opt = optim.Adam(learning_rate=1e-5)
#create metrics class to measure fine-tuning progress
class Metrics:
    train_losses = []
    val_losses = []
    def on_train_loss_report(self, info):
        self.train_losses.append((info["iteration"], info["train_loss"]))
    def on_val_loss_report(self, info):
        self.val_losses.append((info["iteration"], info["val_loss"]))
        
metrics = Metrics()


# %%
# start fine-tuning
start_time = time.time()

train(
    model = model,
    tokenizer = tokenizer,
    args = training_args,
    optimizer = opt,
    train_dataset = train_set,
    val_dataset = dev_set,
    training_callback = metrics
)
end_time = time.time()
duration = end_time - start_time
print(f"Completed finetuning Training in {duration/60:.2f} minutes")

# plot graph of fine-tuning
train_its, train_losses = zip(*metrics.train_losses)
val_its, val_losses = zip(*metrics.val_losses)
# Plot the training and validation losses
# to-do: plt.plot gives an error
plt.plot(train_its, train_losses, '-o')
plt.plot(val_its, val_losses, '-o')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend(['Train', "Valid"])
plt.show() 
plt.savefig("finetuning_loss.png")
# Build Lora model 
model_lora, _ = load("meta-llama/Llama-3.2-1B-Instruct", 
                        adapter_path=adapter_path,)
print("Executing finetuned model")
print(prompt)
response = generate(model_lora, tokenizer, prompt=prompt, verbose=True)
print(response)

