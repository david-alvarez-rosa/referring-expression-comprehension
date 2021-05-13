"""Code for the Speech to Text (STT) task.

Using Silero model.
"""

import argparse
import torch
import zipfile
import torchaudio
from glob import glob


device = torch.device('cpu')

model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device,
                                       verbose=False)

(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils

parser = argparse.ArgumentParser(description="ArgumentParser")
parser.add_argument("--file", help="Name of audio file", required=True)

args = parser.parse_args()

test_files = glob(args.file)
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]),
                            device=device)

output = model(input)
for example in output:
    print(decoder(example.cpu()))
