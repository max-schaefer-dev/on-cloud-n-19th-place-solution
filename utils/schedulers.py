import torch
import matplotlib.pyplot as plt

def get_lr_scheduler(opt, scheduler_name):

    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    else:
        raise NotImplemented

    return scheduler


def save_lr_scheduler_as_jpg(epochs, output_dir):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    lrs = []

    for i in range(epochs):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.plot(lrs)
    plt.xlabel('epoch'); plt.ylabel('learnig rate')
    plt.title('Learning Rate Scheduler')
    plt.savefig(f'{output_dir}/lr_schedule.jpg',dpi=300,bbox_inches='tight',pad_inches=0)