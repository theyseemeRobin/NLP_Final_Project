import argparse
from datetime import datetime
import optuna
from optuna.pruners import HyperbandPruner
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, RobertaForQuestionAnswering
import yaml

from dataprocessing import SquadData
from my_util import *

from train import train_classifier


parser = argparse.ArgumentParser(description="RoBERTa fine tuning for emotion classification.")
run_id = datetime.now().strftime('%b_%d_%H_%M_%S')
parser.add_argument("--tune", action="store_true",
                    help="Runs the main script using optuna to tune hyperparameters. Note that this means a different "
                         "config file that specifies ranges has to be used")
parser.add_argument("--run-id", type=str, default=run_id,
                    help="Name of the current run (results are stored at data/results/<run-id>)")
parser.add_argument("--train-data", type=str, default="data/raw/train.json",
                    help="Path of the JSON file that contains the training data")
parser.add_argument("--test-data", type=str, default="data/raw/test.json",
                    help="Path of the JSON file that contains the test data")
parser.add_argument("--config", type=str, default="data/config.yaml",
                    help="Path of the file that contains the hyperparameters or their tuning configurations.")
args = parser.parse_args()


def config_from_trial(hyperparameters, trial):
    """
    Parameters
    ----------
    trial : optuna.Trial
    hyperparameters : dict
    """
    run_config = hyperparameters['set']
    for hp, hp_range in hyperparameters['tunable'].items():
        if hp_range['type'] == "float":
            run_config[hp] = trial.suggest_float(hp, hp_range['min'], hp_range['max'], log=hp_range['log'])
        elif hp_range['type'] == "int":
            run_config[hp] = trial.suggest_int(hp, hp_range['min'], hp_range['max'], log=hp_range['log'])
    return run_config


def main(run_config, save_dir, trial=None):
    """
    Runs a RoBERTa fine-tuning experiment.

    Parameters
    ----------
    run_config : dict
        Dictionary that contains the hyperparameters, or the tuning information when tuning.
    trial : optuna.Trial
    save_dir : str
        Path where the models and results are saved
    Returns
    -------
    tuple :
        weighted F1 when tuning hyperparameters, train/test LossCurves and a Metrics object otherwise.
    """
    tensorboard_dir = os.path.join(save_dir, "tensorboard")
    # Get suggested HP's through optuna when tuning
    tuning = False
    tag = "Final"
    if trial is not None:
        tuning = True
        run_config = config_from_trial(run_config, trial)
        tag = trial.number

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Note that the training set is split when tuning, while the test set is used otherwise
    tokenizer = AutoTokenizer.from_pretrained(run_config['model'])
    dataset = SquadData(args.train_data, tokenizer, run_config['context_window'], device=device)
    if tuning:
        training_data, validation_data = torch.utils.data.random_split(dataset, [0.7, 0.3])
        train_loader = DataLoader(training_data, batch_size=run_config['batch_size'], shuffle=True)
        eval_loader = DataLoader(validation_data, batch_size=run_config['test_batch_size'], shuffle=True)
    else:
        test_data = SquadData(args.test_data, tokenizer, run_config['context_window'], device=device)
        train_loader = DataLoader(dataset, batch_size=run_config['batch_size'], shuffle=True)
        eval_loader = DataLoader(test_data, batch_size=run_config['test_batch_size'], shuffle=True)

    model = RobertaForQuestionAnswering.from_pretrained(run_config['model']).to(device=device)

    # Freeze pre-trained weights
    if run_config['freeze_weights']:
        for param in model.roberta.parameters():
            param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=run_config['learning_rate'], weight_decay=run_config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader) * run_config['n_epochs'] * run_config['warmup_fraction']),
        num_training_steps=len(train_loader) * run_config['n_epochs']
    )
    training_losses, test_losses, metrics = train_classifier(
        model,
        run_config,
        optimizer,
        scheduler,
        train_loader,
        eval_loader,
        tokenizer,
        trial=trial,
        tensorboard_dir=os.path.join(tensorboard_dir, f"trial {tag}"),
        save_dir=os.path.join(save_dir, "models", f"trial {tag}") if save_dir is not None else None,
        device=device
    )
    if tuning:
        return metrics[-1].f1_score
    else:
        return MakeCurve(mean=training_losses, tag="Train"), MakeCurve(mean=test_losses, tag="Test"), metrics


if __name__ == "__main__":

    base_output_path = os.path.join("data", "results", f"{args.run_id}")
    os.makedirs(base_output_path)
    print(f"Saving results to: {base_output_path}")

    # Find the best hyperparameters through optuna
    if args.tune:
        copy_file(args.config, os.path.join(base_output_path))
        with open(args.config, 'r') as file:
            hp_ranges = yaml.safe_load(file)
        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///data/results/" + args.run_id + "/tuning.db",
            study_name="RoBERTa tuning",
            load_if_exists=True,
            pruner=HyperbandPruner()
        )
        study.optimize(lambda trial: main(hp_ranges, trial=trial, save_dir=base_output_path), n_trials=hp_ranges['n_trials'])
        best_trial = study.best_trial
        config = best_trial.params
        config.update(hp_ranges['set'])

    # Or use the hyperparameters specified in the config file
    else:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    # Write the config to the output directory (so we can see which HP's are used later)
    with open(os.path.join(base_output_path, "hyperparameters.yaml"), 'w') as file:
        yaml.dump(config, file)

    # Do a final training run with the selected HP's
    train_curve, test_curve, metrics = main(
        config,
        save_dir=os.path.join(base_output_path, "models"),
    )

    # Plot results and show F1/EM scores
    plot_curves([train_curve, test_curve], base_output_path, "LossCurve", save_data=True)
    print(f"F1-score: {metrics[-1].f1_score}")
    print(f"EM-score: {metrics[-1].f1_score}")
