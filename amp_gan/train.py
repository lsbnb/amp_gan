import sys
import os
import argparse
import torch
from torch import autograd

sys.path.append(os.path.dirname(__file__) + "/..")
from utils import create_folder, get_conversion_table, read_fasta
from utils import get_encoded_seqs, generate_seqs, get_batch_simple_identity
from utils import plot_loss, plot_identity, write_fasta
from model import get_model_and_optimizer


def calculate_gradient_penalty(discriminator, real_data, fake_data, lambda_value=None):
    lambda_value = lambda_value or 10
    alpha = torch.rand(len(real_data), 1)
    alpha = alpha.expand(real_data.permute(2, 3, 0, 1).size())
    fake_data = fake_data.permute(2, 3, 0, 1)
    real_data = real_data.permute(2, 3, 0, 1)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    interpolates = interpolates.permute(2, 3, 0, 1)
    interpolated_score = discriminator(interpolates)
    grad_outputs = torch.ones(interpolated_score.size())
    gradients = autograd.grad(
        outputs=interpolated_score,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradient_penalty = ((gradients[0].norm(2, dim=1) - 1) ** 2).mean() * lambda_value
    return gradient_penalty


def train(seqs, table, batch_size, latent_size, num_epoch, identity_step):
    returned = get_model_and_optimizer(latent_size, 6, 64)
    generator, discriminator, gen_optim, discrim_optim = returned
    encoded = get_encoded_seqs(seqs, table)
    dataset = torch.utils.data.TensorDataset(torch.Tensor(encoded))
    gen_losses = []
    discrim_losses = []
    identities = []
    collected_seqs = {}
    print("Starting training loop...")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    noise = torch.randn(batch_size, latent_size, 1, 1)
    for epoch in range(1, 1 + num_epoch):
        print("epoch", epoch)
        for i, data in enumerate(dataloader):
            discriminator.zero_grad()
            real_seqs = data[0]
            real_disc_val = discriminator(real_seqs).view(-1)
            noise = torch.randn(batch_size, latent_size, 1, 1)
            fake_seqs = generator(noise)
            fake_disc_val = discriminator(fake_seqs.detach()).view(-1)
            gp = calculate_gradient_penalty(discriminator, real_seqs, fake_seqs)
            discrim_loss = -torch.mean(real_disc_val) + torch.mean(fake_disc_val) + gp
            discrim_loss.backward()
            discrim_optim.step()
            discrim_losses.append(discrim_loss.item())
            if i % 5 == 0:
                generator.zero_grad()
                fake_disc_val = discriminator(fake_seqs).view(-1)
                gen_loss = -torch.mean(fake_disc_val)
                gen_loss.backward()
                gen_optim.step()
                gen_losses.append(gen_loss.item())
            if epoch % identity_step == 0 and i == len(dataloader) - 1:
                with torch.no_grad():
                    generated_seqs = generate_seqs(generator, table, noise)
                    collected_seqs[epoch] = generated_seqs
                    identities.append(get_batch_simple_identity(generated_seqs, seqs))
    return collected_seqs, identities, gen_losses, discrim_losses, generator


def main(fasta_path, output_root, batch_size=None, epoch=None, step=None):
    create_folder(output_root)
    batch_size = batch_size or 128
    epoch = epoch or 10000
    step = step or 100
    latent_size = 100
    table_path = os.path.join(os.path.dirname(__file__), "physical_chemical_6.txt")
    table = get_conversion_table(table_path)
    seqs = read_fasta(fasta_path)
    selected_seqs = {}
    for id_, seq in seqs.items():
        if len(seq) < 30:
            selected_seqs[id_] = seq
    returned = train(selected_seqs, table, batch_size, latent_size, epoch, step)
    collected_seqs, identities, gen_losses, discrim_losses, generator = returned
    plot_loss(gen_losses, discrim_losses, os.path.join(output_root, "loss_figure.png"))
    if len(identities) > 0:
        path = os.path.join(output_root, "identity_step.png")
        plot_identity(identities, step, path)
    for i, seqs in collected_seqs.items():
        path = os.path.join(output_root, "epoch_{}_generated_seq.fasta".format(i))
        write_fasta(seqs, path)
    torch.save(generator, os.path.join(output_root, "generator.pkl"))
    noise = torch.randn(len(selected_seqs), latent_size, 1, 1)
    generated_seqs = generate_seqs(generator, table, noise)
    write_fasta(generated_seqs, os.path.join(output_root, "final_generated_seq.fasta"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta_path", required=True)
    parser.add_argument("-o", "--output_root", required=True)
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-e", "--epoch", type=int)
    parser.add_argument("-s", "--step", type=int)
    args = vars(parser.parse_args())
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    main(**args)
