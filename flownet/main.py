import argparse

from flownet.dataset.flow_dataloader import get_sintel_flow_dataloader
from flownet.network_trainer import NetworkTrainer
from flownet.utils.torch_utils import set_random_seed, check_device


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", "--experiment_name", required=True, help=r"Name of experiment")
    parser.add_argument("-e", "--epochs", type=int, required=True, help=r"Number of epochs to be trained on.")
    parser.add_argument("-batch", "--batch_size", type=int, required=False, default=4,
                        help="How many samples to be consider in a training mini batch.")
    parser.add_argument("-gpu", "--gpu_number", type=int, required=True, default=0, help="GPU used for training/evaluation")
    parser.add_argument("-lr", "--learning_rate", type=float, required=False, default="1e-4",
                        help="Values of the learning rate for optimizer.")
    parser.add_argument("-es", "--early_stopping", required=False, default=False, action="store_true",
                        help="Whether to use early stopping or not")
    parser.add_argument("-patience", "--patience_coeff", type=int, required=False, default=5)
    parser.add_argument("-data", "--dataset_path", required=True, help="Path to dataset root dir.")
    parser.add_argument("-out", "--output_path", required=True, help="Path where results will be saved.")
    parser.add_argument("-t", "--train", required=False, action="store_true", help="Training mode")
    parser.add_argument("-resume", "--resume_training", required=False, default=False, action="store_true",
                        help="If true, resume training from checkpoint; otherwise, start from beginning")
    parser.add_argument("-chk", "--checkpoint_path", required=False, default="", help="Checkpoint from which to resume training.")
    parser.add_argument("-visu", "--training_visu", required=False, default=False, action="store_true",
                        help="If true, create visus during training")
    parser.add_argument("-ov", "--overfit", required=False, default=False, action="store_true",
                        help="If true, train to overfit a single batch of data")
    return parser.parse_args()


def main(args):
    set_random_seed(0)
    device = check_device(args.gpu_number)

    train_dataloader, validation_dataloader, test_dataloader = get_sintel_flow_dataloader(args, batch_overfit_mode=args.overfit)

    trainer = NetworkTrainer(args.output_path, args.experiment_name, checkpoint_path=args.checkpoint_path)

    if args.train:
        trainer.train_model(train_dataloader, validation_dataloader, device, args)
    trainer.evaluate_model(test_dataloader, device, args)


if __name__ == '__main__':
    args = parse_cmd()
    main(args)
