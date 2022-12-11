import os
import torch
from torch.nn import functional as F
from dataset import return_MVTecAD_loadertrain, return_MVTecAD_loadertest
from network import VAE, loss_function
import matplotlib.pyplot as plt


def train(model, train_loader, device, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data, model.mu, model.logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    return train_loss


def eval(model, test_loader, device):
    model.eval()
    x_0 = next(iter(test_loader))
    with torch.no_grad():
        x_vae = model(x_0.to(device)).detach().cpu().numpy()


def EBM(model, test_loader, device):
    model.train()
    x_0 = next(iter(test_loader))
    alpha = 0.05
    lamda = 1
    x_0 = x_0.to(device).clone().detach().requires_grad_(True)
    recon_x = model(x_0).detach()
    loss = F.binary_cross_entropy(x_0, recon_x, reduction='sum')
    loss.backward(retain_graph=True)

    x_grad = x_0.grad.data
    x_t = x_0 - alpha * x_grad * (x_0 - recon_x) ** 2

    for i in range(15):
        recon_x = model(x_t).detach()
        loss = F.binary_cross_entropy(x_t, recon_x, reduction='sum') + lamda * torch.abs(x_t - x_0).sum()
        loss.backward(retain_graph=True)

        x_grad = x_0.grad.data
        eps = 0.001
        x_t = x_t - eps * x_grad * (x_t - recon_x) ** 2
        iterative_plot(x_t.detach().cpu().numpy(), i)


# gif
def iterative_plot(x_t, j):
    plt.figure(figsize=(15, 4))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_t[i][0], cmap=plt.cm.gray)
    plt.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig("/Users/ryotamarumoto/Work/corpy/models/AE/AE5/results/{}.png".format(j))
    # plt.show()


def main():
    train_loader = return_MVTecAD_loadertrain(image_dir="/Users/ryotamarumoto/Work/corpy/archive/dataset/train/good", batch_size=256, train=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    seed = 42
    out_dir = '/Users/ryotamarumoto/Work/corpy/models/AE/AE5/logs'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    checkpoints_dir = "/Users/ryotamarumoto/Work/corpy/models/AE/AE5/checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = VAE(z_dim=512).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    num_epochs = 50
    for epoch in range(num_epochs):
        loss = train(model=model, train_loader=train_loader, device=device, optimizer=optimizer, epoch=epoch)
        print('epoch [{}/{}], train loss: {:.4f}'.format(epoch + 1, num_epochs, loss))
        # if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, "{}.pth".format(epoch + 1)))
    test_loader = return_MVTecAD_loadertest(image_dir="/Users/ryotamarumoto/Work/corpy/archive/dataset/test/", batch_size=10, train=False)
    eval(model=model, test_loader=test_loader, device=device)
    EBM(model, test_loader, device)


if __name__ == "__main__":
    main()