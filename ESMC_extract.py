import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

import pathlib
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

output_dir = pathlib.Path('../data/ESMC-300M/')
df_seq = pd.read_csv('../data/test.csv', header=None)
seqs = df_seq.iloc[:, 1].values.tolist()
names = df_seq.iloc[:, 0].values.tolist()

client = ESMC.from_pretrained('esmc_300m').to(device)

for seq, name in zip(seqs, names):
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))

    emb = logits_output.embeddings.squeeze()
    output_file = (output_dir / f"{name}.pt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, output_file)
    print('save embed files for ', name)
