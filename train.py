from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import utils
import visual


def train(model, dataset,
          lr=1e-04, weight_decay=1e-04, beta1=0.5, beta2=.999, lamda=10.,
          batch_size=32, sample_size=32, epochs=10,
          d_trains_per_g_train=2,
          checkpoint_dir='checkpoints',
          checkpoint_interval=1000,
          image_log_interval=100,
          loss_log_interval=30,
          resume=False, cuda=False):
    # define the optimizers.
    generator_optimizer = optim.Adam(
        model.generator.parameters(), lr=lr, betas=(beta1, beta2),
        weight_decay=weight_decay
    )
    critic_optimizer = optim.Adam(
        model.critic.parameters(), lr=lr, betas=(beta1, beta2),
        weight_decay=weight_decay
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
            d_trains = (
                30 if (batch_index < 25 or batch_index % 500 == 0) else
                d_trains_per_g_train
            )

            # run the critic and backpropagate the errors.
            for _ in range(d_trains):
                critic_optimizer.zero_grad()
                z = model.sample_noise(batch_size)
                c_loss, g = model.c_loss(x, z, return_g=True)
                c_loss_gp = c_loss + model.gradient_penalty(x, g, lamda=lamda)
                c_loss_gp.backward()
                critic_optimizer.step()

            # run the generator and backpropagate the errors.
            generator_optimizer.zero_grad()
            z = model.sample_noise(batch_size)
            g_loss = model.g_loss(z)
            g_loss.backward()
            generator_optimizer.step()

            # update the progress.
            data_stream.set_description((
                'epoch: {epoch}/{epochs} |'
                'iteration: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'g: {g_loss:.4} / '
                'w: {w_dist:.4}'
            ).format(
                epoch=epoch,
                epochs=epochs,
                iteration=iteration,
                trained=batch_index*batch_size,
                total=dataset_size,
                progress=(100.*batch_index/dataset_batches),
                g_loss=g_loss.data[0],
                w_dist=-c_loss.data[0],
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
                    model.sample_image(sample_size).data,
                    'generated samples',
                    env=model.name
                )

            # save the model at checkpoints.
            if iteration % checkpoint_interval == 0:
                # notify that we've reached to a new checkpoint.
                print()
                print()
                print('#############')
                print('# checkpoint!')
                print('#############')
                print()

                utils.save_checkpoint(model, checkpoint_dir, iteration)

                print()
