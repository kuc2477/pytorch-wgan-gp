from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import utils
import visual


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
        iteration = utils.load_checkpoint(model, checkpoint_dir)
        epoch_start = iteration // (len(dataset) // batch_size) + 1

    for epoch in range(epoch_start, epochs+1):
        data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (x, _) in data_stream:
            # where are we?
            dataset_size = len(data_loader.dataset)
            dataset_batches = len(data_loader)
            iteration = (
                (epoch-1)*(dataset_size // batch_size) +
                batch_index + 1
            )

            # prepare the data.
            x = Variable(x).cuda() if cuda else Variable(x)

            # run the critic and backpropagate the errors.
            for _ in range(d_trains_per_g_train):
                critic_optimizer.zero_grad()
                z = model.sample_noise(batch_size)
                c_loss, g = model(z, x, retrieve_generated_images=True)
                c_loss += model.gradient_penalty(x, g, lamda=lamda)
                c_loss.backward()
                critic_optimizer.step()

            # run the generator and backpropagate the errors.
            generator_optimizer.zero_grad()
            g_loss = model(z)
            g_loss.backward()
            generator_optimizer.step()

            # update the progress.
            data_stream.set_description((
                'epoch: {epoch} |'
                'iteration: {iteration} |'
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'g: {g_loss:.4} / '
                'w: {w_dist:.4} / '
            ).format(
                epoch=epoch,
                iteration=iteration,
                trained=batch_index*batch_size,
                total=dataset_size,
                progress=(100.*batch_index/dataset_batches),
                g_loss=g_loss,
                w_dist=-c_loss,
            ))

            # send losses to the visdom server.
            if iteration % loss_log_interval == 0:
                visual.visualize_scalar(
                    -c_loss.data[0],
                    'estimated wasserstein distance between x and g',
                    iteration=iteration,
                    env=model.name
                )
                visual.visualize_scalar(
                    g_loss.data[0],
                    'generator loss',
                    iteration=iteration,
                    env=model.name
                )

            # send sample images to the visdom server.
            if iteration % image_log_interval == 0:
                visual.visualize_images(
                    model.sample_image(32),
                    'generated samples',
                    env=model.name
                )

            # save the model at checkpoints.
            if iteration % checkpoint_interval == 0:
                utils.save_checkpoint(model, checkpoint_dir, iteration)
