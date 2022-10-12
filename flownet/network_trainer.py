import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from flownet.dataset.flow_datasets import SINTEL_IMG_SIZE_CROPPED
from flownet.utils.io_utils import save_dict_json
from flownet.model.losses import EPE
from flownet.model.model import FlowNetCorr
from flownet.utils.logger import create_logger
from flownet.visu.visu import predictions_visu


class NetworkTrainer:
    def __init__(self, output_dir, experiment_name, checkpoint_path: str = ""):
        self.output_path = os.path.join(output_dir, experiment_name)
        os.makedirs(self.output_path, exist_ok=True)

        self.visu_path = os.path.join(self.output_path, "visu")
        os.makedirs(self.visu_path, exist_ok=True)

        if len(checkpoint_path) == 0:
            self.checkpoint_path = os.path.join(self.output_path, "model_checkpoint.pth")
        else:
            self.checkpoint_path = checkpoint_path

        # Logging
        self.logger = create_logger("experiment", file_handler_path=self.output_path)
        self.tb_writer = SummaryWriter(os.path.join(output_dir, "runs", experiment_name))

    def train_model(self, train_dataloader, validation_dataloader, device, args):
        # Init model
        model = FlowNetCorr().to(device)
        input_size = (args.batch_size, 3, SINTEL_IMG_SIZE_CROPPED[0], SINTEL_IMG_SIZE_CROPPED[1])
        summary(model, input_size=(input_size, input_size))

        # Init optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_func = EPE

        # if args.early_stopping:
            # TODO: use pytorch implementation
            # early_stopping = EarlyStopping(args.patience_coeff, path=self.checkpoint_path, verbose=True)
        # else:
        early_stopping = None

        # Save training config
        self.save_training_config(vars(args), model, optimizer)

        # Training loop
        for e in range(1, int(args.epochs) + 1):
            self.logger.info(f"Epoch {e}\n-------------------------------")
            create_visu = args.training_visu and e % 10 == 0

            train_stats = self.train_loop(train_dataloader, model, loss_func, optimizer, device)
            val_stats = self.test_loop(validation_dataloader, model, loss_func, device, mode="val", create_visu=create_visu, epoch=e)

            if early_stopping is not None:
                early_stopping(val_stats["val_loss"], model)
                if early_stopping.early_stop:
                    self.logger.info(f"Early stopping after epoch {e}")
                    break

            self.write_to_tensorboard(e, train_stats, val_stats)

        torch.save(model.state_dict(), self.checkpoint_path)

        self.logger.info("Done!")
        self.tb_writer.flush()
        self.tb_writer.close()

    def evaluate_model(self, test_dataloader, device, args):
        if len(self.checkpoint_path) == 0:
            self.logger.info(f"No checkpoint found")
            return

        visu_path = os.path.join(self.output_path, "visu")
        os.makedirs(visu_path, exist_ok=True)

        model = FlowNetCorr().to(device)
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=device))
        model.eval()
        self.logger.info(f"Loaded checkpoint: {self.checkpoint_path}")

        loss_func = EPE

        test_stats, _ = self.test_loop(test_dataloader, model, loss_func, device, mode="test", create_visu=True)
        save_dict_json(test_stats, filename="test_set_stats", output_dir=self.output_path)
        self.logger.info("Saved test statistics")

    def train_loop(self, dataloader, model, loss_func, optimizer, device):
        dataset_size = len(dataloader.dataset)
        num_batches = len(dataloader)
        mean_loss = 0

        for batch_id, batch_data in enumerate(dataloader):
            img1 = batch_data["img1"].to(device)
            img2 = batch_data["img2"].to(device)
            gt_flow = batch_data["flow"].to(device)

            # Convert images from B x H x W x C to B x C x H x W
            img1 = img1.permute((0, 3, 1, 2)).float()
            img2 = img2.permute((0, 3, 1, 2)).float()
            gt_flow = gt_flow.permute((0, 3, 1, 2)).float()

            # Compute prediction and loss
            pred_flow = model(img1, img2)
            loss = loss_func(pred_flow, gt_flow)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 5 == 0:
                loss_val, epoch_progress = loss.item(), batch_id * len(img1)
                self.logger.info(f"Training loss (batch): {loss_val:>7f}  [{epoch_progress:>5d}/{dataset_size:>5d}]")

            # Training statistics
            mean_loss += loss.item()

        mean_loss /= num_batches
        self.logger.info(f"[train] Loss: {mean_loss:>8f};")
        return {"train_loss": mean_loss}

    def test_loop(self, dataloader, model, loss_func, device, mode="test", create_visu=False, epoch=None):
        num_batches = len(dataloader)
        mean_loss = 0

        with torch.no_grad():
            for batch_id, batch_data in enumerate(dataloader):
                img1 = batch_data["img1"].to(device)
                img2 = batch_data["img2"].to(device)
                gt_flow = batch_data["flow"].to(device)

                # Convert images from B x H x W x C to B x C x H x W
                img1 = img1.permute((0, 3, 1, 2)).float()
                img2 = img2.permute((0, 3, 1, 2)).float()
                gt_flow = gt_flow.permute((0, 3, 1, 2)).float()

                pred_flow = model(img1, img2)
                loss = loss_func(pred_flow, gt_flow)

                mean_loss += loss.item()

                if create_visu:
                    img1 = img1.cpu().numpy()
                    img2 = img2.cpu().numpy()
                    gt_flow = gt_flow.cpu().numpy()
                    pred_flow = pred_flow.cpu().numpy()
                    additional_info = f"result_epoch_{epoch}" if epoch is not None else "result"

                    for i in range(dataloader.batch_size):
                        sample_id = batch_id * dataloader.batch_size + i
                        predictions_visu(img1[i, :, :, :], img2[i, :, :, :], gt_flow[i, :, :, :], pred_flow[i, :, :, :],
                                         sample_id, output_path=self.visu_path, additional_info=additional_info)

        mean_loss /= num_batches
        self.logger.info(f"[{mode}] Loss: {mean_loss:>8f}; ")
        return {f"{mode}_loss": mean_loss}

    def save_training_config(self, args, model, optimizer):
        training_config = {"model": type(model).__name__, "optimizer": type(optimizer).__name__}
        training_config = {**args, **training_config}

        save_dict_json(training_config, filename=f"training_config", output_dir=self.output_path)
        self.logger.info("Training config:")
        for key in training_config.keys():
            self.logger.info(f"{key}: {training_config[key]}")

    def write_to_tensorboard(self, epoch, train_stats, val_stats):
        self.tb_writer.add_scalar("Loss/train", train_stats["train_loss"], epoch)
        self.tb_writer.add_scalar("Loss/val", val_stats["val_loss"], epoch)
