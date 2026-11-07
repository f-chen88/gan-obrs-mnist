import torch
import os

from obrs import find_c, density_ratio

def critic(u):
    return u * torch.log(u) - (1+u) * torch.log(1+u)

def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def G_train_obrs(x, G, D, G_optimizer, budget, device, tolerance = 1e-4, n_iter = 100):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
                 
    G_output = G(z)
    r = density_ratio(D, G_output)
    c = find_c(r, budget, tol=tolerance, n_iter=n_iter)
    acceptance_ratios = torch.clamp(c * r, max=1.0)
    G_loss = torch.mean(acceptance_ratios* budget * critic(r / (acceptance_ratios * budget)))

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder, device, name='G.pth'):
    ckpt_path = os.path.join(folder, name)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def search_optimal_threshold(likelihood_ratios, budget = 2.0, precision = 1e-5):
    """
    Perform a dichotomic search to find the optimal threshold for OBRS.
    """
    low = 1e-10
    high = 1e10
    optimal_threshold = low + high / 2
    def loss_function(threshold):
        acceptance_ratios = likelihood_ratios * threshold
        acceptance_ratios = torch.clamp(acceptance_ratios, max=1.0)
        return torch.mean(acceptance_ratios) - 1 / budget
    loss = loss_function(optimal_threshold)
    while abs(loss) >= precision:
        if loss > precision:
            high = optimal_threshold
        elif loss < -precision:
            low = optimal_threshold
        optimal_threshold = (low + high) / 2
        loss = loss_function(optimal_threshold)
    return optimal_threshold
