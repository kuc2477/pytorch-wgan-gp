from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm
import utils


def train(model, dataset, checkpoint_dir='checkpoints',
          lr=1e-04, lr_decay=1e-04, lamda=10.,
          batch_size=32, sample_size=32, epochs=10,
          d_trains_per_g_train=2,
          checkpoint_interval=1000,
          image_log_interval=100,
          loss_log_interval=30,
          resume=False, cuda=False):
    # define the optimizers.
    generator_optimizer = optim.Adam(
        model.generator.parameters(), lr=lr,
        weight_decay=lr_decay
    )
    critic_optimizer = optim.Adam(
        model.critic.parameters(), lr=lr,
        weight_decay=lr_decay
    )

    # prepare the model and statistics.
    model.train()
    epoch_start = 1

    # load checkpoint if needed.
    if resume:
        iteration = utils.load_checkpoint(model, model_dir)
        epoch_start = iteration // (len(dataset) // batch_size) + 1

    for epoch in range(epoch_start, epochs+1):
        data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (x, _) in data_stream:
            # where are we?
            data_size = len(x)
            dataset_size = len(data_loader.dataset)
            dataset_batches = len(data_loader)
            iteration = (
                (epoch-1)*(dataset_size // batch_size) +
                batch_index + 1
            )

            x = Variable(x).cuda() if cuda else Variable(x)
