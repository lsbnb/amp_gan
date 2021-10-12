import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.nn.functional import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from Bio import SeqIO


def get_conversion_table(path, norm=True):
    table = pd.read_csv(path, sep=" ", index_col=0)
    index = list(table.index)
    if norm:
        scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(table)
        table = {}
        for index, aa in enumerate(index):
            table[aa] = np.array(scaled[index])
    table["X"] = [0] * 6
    return table


def read_fasta(fasta_path):
    fasta = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        fasta[str(record.id)] = str(record.seq)
    return fasta


def write_fasta(seqs, path):
    with open(path, "w") as output:
        for name, seq in seqs.items():
            output.write(">{}\n{}\n".format(name, seq))


def padding_seqs(seqs, length=None, pad_value=None):
    length = length or 30
    pad_value = pad_value or "X"
    data = {}
    for key, seq in seqs.items():
        if len(seq) < length:
            data[key] = seq + pad_value * (length - len(seq))
        else:
            raise Exception("Length exceeds {}".format(length))
    return data


def encode(fasta, table):
    encoded_seqs = {}
    for key, seq in fasta.items():
        encoded_seqs[key] = [table[aa] for aa in list(seq)]
    return encoded_seqs


def get_encoded_seqs(seqs, table):
    seqs = padding_seqs(seqs)
    encoded_seqs = encode(seqs, table)
    encoded_seqs = np.array(list(encoded_seqs.values())).astype(float)
    encoded_seqs = np.expand_dims(encoded_seqs, axis=1)
    return encoded_seqs


def decode(seqs, table):
    decode_seqs = []
    for seq in seqs:
        seq = seq.reshape(30, -1)
        decode_seq = ""
        for index in range(seq.shape[0]):
            generated_vector = seq[index]
            similarity = {}
            for key, aa_vector in table.items():
                aa_vector = torch.FloatTensor(aa_vector)
                score = cosine_similarity(generated_vector, aa_vector, dim=-1, eps=1e-8)
                similarity[key] = score
            key_max = max(similarity.keys(), key=(lambda k: similarity[k]))
            decode_seq += key_max
        decode_seqs.append(decode_seq)
    return decode_seqs


def generate_seqs(net, table, noise):
    generated = net(noise)
    decoded_seqs = decode(generated, table)
    seqs = {}
    for i, decoded_seq in enumerate(decoded_seqs):
        seq = [aa for aa in decoded_seq if aa != "X"]
        seqs["seq_{}".format(i + 1)] = "".join(seq)
    return seqs


def get_simple_identity(seq_1, seq_2):
    i = 0
    for aa_1, aa_2 in zip(seq_1, seq_2):
        if aa_1 == aa_2:
            i = i + 1
    min_lenth = min(len(seq_1), len(seq_2))
    identity = i / min_lenth
    return identity


def get_batch_simple_identity(fake_seqs, real_seqs):
    max_identity_list = []
    for fake_seq in fake_seqs:
        identity_list = []
        for real_seq in real_seqs:
            identity_list.append(get_simple_identity(fake_seq, real_seq))
        max_identity_list.append(max(identity_list))
    return max_identity_list


def plot_identity(values, step, path):
    labels = [i * step for i in range(1, 1 + len(values))]
    fig, axs = plt.subplots(figsize=(15, 5))
    axs.set_ylabel("Generated sequence identity to natural ones")
    axs.set_xlabel("Epoch")
    axs.boxplot(np.transpose(values), labels=labels)
    plt.savefig(path)


def plot_loss(gen_losses, discrim_losses, path):
    plt.figure(figsize=(10, 5))
    plt.title("Generator loss and discriminator loss during training")
    plt.plot(gen_losses, label="Generator")
    plt.plot(discrim_losses, label="Discriminator")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)


def create_folder(root):
    if not os.path.exists(root):
        os.makedirs(root)
