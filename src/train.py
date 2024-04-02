import os
from collections import namedtuple

import numpy as np
import optuna
import torch.nn.functional
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, classification_report

from my_util import token_f1_score

Metrics = namedtuple("metrics", ("f1_score", "em_score"))

def train_classifier(
        model,
        config,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        tokenizer,
        trial=None,
        tensorboard_dir=None,
        save_dir=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the RoBERTa classifier. Returns the lists of training and evaluation loss (test or validation loss, depending
    on if we're tuning HP's) as well as a list of Metrics objects which holds various performance metrics.

    Parameters
    ----------
    trial : optuna.Trial
    model : torch.nn.Module
    config : dict
        A dictionary that contains all hyperparameters
    optimizer : torch.optim.Optimizer
    train_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
    save_dir : str
        Directory where the model state_dicts are saved after each epoch (Default is None and means no saving).
    tensorboard_dir : str
        Directory where tensorboard logs are stored (default is None and means no logging).
    device : torch.device
    scheduler :
        The learning rate scheduler

    Returns
    -------
    tuple[list, list, list[Metrics]] :
        A tuple of list of training losses and test losses over time.
    """

    # Create a writer for tensorboard logging
    writer = None
    if tensorboard_dir is not None:
        writer = SummaryWriter(tensorboard_dir)

    # Initialize progressbar
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
    )

    train_batch_losses = []
    train_epoch_losses = []
    test_epoch_losses = []
    metrics = []
    for epoch in range(config['n_epochs']):

        # Training
        with progress:
            model.train()
            training = progress.add_task(f"[blue]Train: Epoch {epoch+1}/{config['n_epochs']}", total=len(train_loader))
            for batch in train_loader:
                model.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_batch_losses.append(loss.detach().cpu().numpy())
                progress.update(training, advance=1)
            train_epoch_losses.append(float(np.mean(train_batch_losses[-len(train_loader):])))

            # Tensorboard logging
            if writer is not None:
                writer.add_scalar(
                    "Epoch Train loss",
                    train_epoch_losses[-1],
                    global_step=epoch
                )

            # Save model at the end of each epoch
            if save_dir is not None:
                os.makedirs(os.path.join(save_dir), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, f"model.pt"))

        # Testing
        loss, metric = eval_classifier(model, test_loader, tokenizer, epoch, device=device, writer=writer)
        test_epoch_losses.append(loss)
        metrics.append(metric)

        if trial is not None:
            trial.report(metric.f1_score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return train_epoch_losses, test_epoch_losses, metrics


def eval_classifier(
        model,
        eval_loader,
        tokenizer,
        epoch,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        writer=None
):
    """
    Evaluates the classifier on the evaluation data (Test or Val. data) and returns the mean loss over the entire epoch,
    and the Metrics object that holds various performance metrics.

    Parameters
    ----------
    epoch :
    writer : torch.utils.tensorboard.SummaryWriter
        The Summary writer that writes tensorboard event files
    model : torch.nn.Module
    eval_loader : torch.utils.data.DataLoader
    device : torch.device
    Returns
    -------
    tuple[float, Metrics]
    """

    # Initialize progressbar
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
    )

    test_batch_losses = []
    pred_answers = []
    true_answers = []
    text_pred_answers = []
    text_true_answers = []
    f1_scores = []
    em_scores = []
    with progress:

        # Testing
        model.eval()
        testing = progress.add_task(f"[red]Testing:", total=len(eval_loader))
        for batch in eval_loader:
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                start_idx = outputs.start_logits.argmax(dim=1).detach().cpu()
                end_idx = outputs.end_logits.argmax(dim=1).detach().cpu()

            test_batch_losses.append(loss.cpu().numpy())
            progress.update(testing, advance=1)

            for idx, (start, end, true_start, true_end) in enumerate(zip(start_idx, end_idx, batch["start_positions"], batch["end_positions"])):
                if end - start < 0 or end < 0 or start < 0:
                    start = end = 0
                text_pred_answers.append(tokenizer.decode(batch["input_ids"][idx][start:end + 1]))
                text_true_answers.append(tokenizer.decode(batch["input_ids"][idx][true_start:true_end + 1]))
                pred_answers.append(batch["input_ids"][idx][start:end + 1])
                true_answers.append(batch["input_ids"][idx][true_start:true_end + 1])
                f1_scores.append(token_f1_score(pred_answers[-1].cpu().tolist(), true_answers[-1].cpu().tolist()))
                em_scores.append(int(pred_answers[-1].cpu().tolist() == true_answers[-1].cpu().tolist()))

        epoch_loss = np.mean(test_batch_losses[-len(eval_loader):])

        # Tensorboard logging
        if writer is not None:
            writer.add_scalar(
                "Epoch Test loss",
                epoch_loss,
                global_step=epoch
            )
            writer.add_scalar(
                "f1 score",
                np.mean(f1_scores),
                global_step=epoch
            )

            writer.add_scalar(
                "EM score",
                np.mean(em_scores),
                global_step=epoch
            )
    return epoch_loss, Metrics(np.mean(f1_scores), np.mean(em_scores))