import torch
from torch.utils.tensorboard import SummaryWriter


SUM_FREQ = 100
VAL_FREQ = 5000


class Logger:
    """ Class used for logging metrics to TensorBoard. """

    def __init__(self, scheduler, sum_freq=SUM_FREQ, val_freq=VAL_FREQ, debug=False):
        """ Initialize logger.

        Args:
            scheduler (torch.optim.LRScheduler): Learning rate scheduler.
            sum_freq (int): Frequency (number of iterations) for printing training status.
            val_freq (int): Frequency (number of iterations) for evaluating on validation set. 
        """
        self.scheduler = scheduler
        self.sum_freq = sum_freq
        self.val_freq = val_freq
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.debug = debug

    def _print_training_status(self):
        # Print training status
        metrics_data = [self.running_loss[k] / self.sum_freq for k in self.running_loss.keys()]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        print(training_str + metrics_str)

        # Initialize writer, if necessary
        if self.writer is None:
            self.writer = SummaryWriter()

        # Add metric to TensorBoard
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / self.sum_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        """ Accumulate metrics for current step.

        Args:
            metrics (dict): Computed metrics for current step/batch.
        """

        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.sum_freq == self.sum_freq - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

    def print_summary_statistics(self, x, name):
        with torch.no_grad():
            min_x = torch.min(x)
            max_x = torch.max(x)
            median_x = torch.median(x)
            std_x, mean_x = torch.std_mean(x)
            print(f"Stats for {name}: min={min_x.item()}, median={median_x.item()}, max={max_x.item()}, mean={mean_x.item()}, std={std_x.item()}")

    def debug_log(self, flow_predictions, flow_gt):
        if not self.debug:
            return

        if self.total_steps % self.sum_freq == self.sum_freq - 1:
            with torch.no_grad():
                u_flow, v_flow = flow_gt[:, 0, :, :], flow_gt[:, 1, :, :]
                self.print_summary_statistics(u_flow, f"u_flow_gt")
                self.print_summary_statistics(v_flow, f"v_flow_gt")

                for idx, flow_pred in enumerate(flow_predictions):
                    if isinstance(flow_pred, tuple):
                        flow, flow_var = flow_pred
                    else:
                        flow, flow_var = flow_pred, None

                    u_flow, v_flow = flow[:, 0, :, :], flow[:, 1, :, :]
                    self.print_summary_statistics(u_flow, f"u_flow_iter_{idx}")
                    self.print_summary_statistics(v_flow, f"v_flow_iter_{idx}")

                    if flow_var is not None:
                        u_flow_var, v_flow_var = flow_var[:, 0, :, :], flow_var[:, 1, :, :]
                        self.print_summary_statistics(u_flow_var, f"u_flow_var_iter_{idx}")
                        self.print_summary_statistics(v_flow_var, f"v_flow_var_iter_{idx}")
