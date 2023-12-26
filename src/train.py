import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """

    if torch.cuda.is_available():
        # Transfer the model to the GPU
        model = model.cuda()

    # Set the model to training mode
    model.train()

    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
            enumerate(train_dataloader),
            desc="Training",
            total=len(train_dataloader),
            leave=True,
            ncols=80,
    ):
        # Move data to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # 1. Clear the gradients of all optimized variables
        optimizer.zero_grad()
        # 2. Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # 3. Calculate the loss
        loss_value = loss(output, target)
        # 4. Backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()
        # 5. Perform a single optimization step (parameter update)
        optimizer.step()

        # Update average training loss
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss))

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    with torch.no_grad():

        # Set the model to evaluation mode
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
                enumerate(valid_dataloader),
                desc="Validating",
                total=len(valid_dataloader),
                leave=True,
                ncols=80,
        ):
            # Move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # 2. Calculate the loss
            loss_value = loss(output, target)

            # Calculate average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss))

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    # Initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a
    # plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss
        )

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # Print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or ((valid_loss_min - valid_loss) / valid_loss_min > 0.01):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }

            # Save the checkpoint
            torch.save(checkpoint, save_path)

            valid_loss_min = valid_loss

        # Update learning rate, i.e., make a step in the learning rate scheduler
        scheduler.step(valid_loss)

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss):
    # Monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():

        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # Move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)
            # 2. Calculate the loss
            loss_value = loss(logits, target)

            # Update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # Convert logits to predicted class
            # The predicted class is the index of the max of the logits
            _, pred = torch.max(logits, 1)

            # Compare predictions to true label
            correct += torch.sum(pred.eq(target.data.view_as(pred))).cpu().item()
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss
