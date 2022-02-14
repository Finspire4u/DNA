import yaml
import os
import torch
import pandas as pd
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from model import Generator, Discriminator
from utils import LPDataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation

config = yaml.load(open('config.yml'))
node_num = config['node_num']
window_size = config['window_size']
G_losses = []
D_losses = []
base_path = os.path.join('./data/', config['dataset'])
train_name = 'train'+'.npy'
train_save_path = os.path.join(base_path, train_name)

train_data = LPDataset(train_save_path, window_size)
sample_data = LPDataset(train_save_path, window_size)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=config['batch_size'],
    shuffle=True,
    pin_memory=True
)
sample_loader = DataLoader(
    dataset=sample_data,
    batch_size=config['batch_size'],
    shuffle=False,
    pin_memory=True
)
generator = Generator(
    window_size=window_size,
    node_num=node_num,
    in_features=config['in_features'],
    out_features=config['out_features'],
    lstm_features=config['lstm_features']
)
discriminator = Discriminator(
    input_size=node_num*node_num*6,
    hidden_size=config['disc_hidden']
)
generator = generator.cuda()
discriminator = discriminator.cuda()
mse = nn.MSELoss(reduction='sum')

generator_optimizer       = optim.RMSprop(generator.parameters(),     lr=config['g_learning_rate'])
discriminator_optimizer   = optim.RMSprop(discriminator.parameters(), lr=config['d_learning_rate'])
# generator_optimizer_2     = optim.RMSprop(generator.parameters(),     lr=config['g_learning_rate_2'])
# discriminator_optimizer_2 = optim.RMSprop(discriminator.parameters(), lr=config['d_learning_rate_2'])
generator_optimizer_3     = optim.RMSprop(generator.parameters(),     lr=config['g_learning_rate_3'])
discriminator_optimizer_3 = optim.RMSprop(discriminator.parameters(), lr=config['d_learning_rate_3'])

print('train GAN')
out = []
for epoch in range(config['gan_epoches']):
    for i, (data, sample) in enumerate(zip(train_loader, sample_loader)):

        # update discriminator
        if epoch < 500:
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
        # elif epoch < 700:
        #     discriminator_optimizer_2.zero_grad()
        #     generator_optimizer_2.zero_grad()
        else:
            discriminator_optimizer_3.zero_grad()
            generator_optimizer_3.zero_grad()
        # discriminator_optimizer.zero_grad()
        # generator_optimizer.zero_grad()

        in_shots, out_shot = data
        in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
        predicted_shot = generator(in_shots)
        fake_logit = discriminator(predicted_shot).mean()

        # update generator
        generator_loss = fake_logit
        generator_loss.backward()
        if epoch < 500:
            generator_optimizer.step()
        # elif epoch < 700:
        #     generator_optimizer_2.step()
        else:
            generator_optimizer_3.step()
        # generator_optimizer.step()

        _, sample = sample
        sample = sample.cuda()
        sample = sample.view(config['batch_size'], -1)
        # print(sample.size())
        real_logit = discriminator(sample).mean()
        fake_logit = discriminator(predicted_shot.detach()).mean()
        discriminator_loss = -real_logit + fake_logit
        discriminator_loss.backward(retain_graph=True)

        if epoch < 500:
            discriminator_optimizer.step()
        # elif epoch < 700:
        #     discriminator_optimizer_2.step()
        else:
            discriminator_optimizer_3.step()
        # discriminator_optimizer.step()

        for p in discriminator.parameters():
            p.data.clamp_(-config['weight_clip'], config['weight_clip'])
        
        out_shot = out_shot.view(config['batch_size'], -1)
        # print(out_shot.size())
        # print(predicted_shot.size())
        # print(predicted_shot)
        out.append(predicted_shot.cpu())
        mse_loss = mse(predicted_shot, out_shot)
        
        # Output training stats
        if (epoch % 50 ==0) & (i % 5 == 0):
            print('[epoch %d] [step %d] [d_loss %.4f] [g_loss %.4f] [mse_loss %.4f]' % (epoch, i,
                discriminator_loss.item(), generator_loss.item(), mse_loss.item()))

        # Save Losses for plotting later
        G_losses.append(generator_loss.item())
        D_losses.append(discriminator_loss.item())

output = pd.DataFrame(out)
output.to_csv('Prediction.csv')

torch.save(generator, os.path.join(base_path, 'generator.pkl'))
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()