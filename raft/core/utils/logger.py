from torch.utils.tensorboard import SummaryWriter

SUM_FREQ = 100
VAL_FREQ = 5000


class Logger:
    """ Class used for logging metrics to TensorBoard. """

    def __init__(self, scheduler, sum_freq=SUM_FREQ, val_freq=VAL_FREQ):
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
