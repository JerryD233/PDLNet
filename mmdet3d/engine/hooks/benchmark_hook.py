# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.hooks import Hook, CheckpointHook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class BenchmarkHook(CheckpointHook):
    """A hook that logs the training speed of each epch."""

    # priority = 'NORMAL'

    # def after_train_epoch(self, runner) -> None:
    #     """We use the average throughput in iterations of the entire training
    #     run and skip the first 50 iterations of each epoch to skip GPU warmup
    #     time.

    #     Args:
    #         runner (Runner): The runner of the training process.
    #     """
    #     message_hub = runner.message_hub
    #     max_iter_num = len(runner.train_dataloader)
    #     speed = message_hub.get_scalar('train/time').mean(max_iter_num - 50)
    #     message_hub.update_scalar('train/speed', speed)
    #     runner.logger.info(
    #         f'Training speed of epoch {runner.epoch + 1} is {speed} s/iter')

    # def after_train(self, runner) -> None:
    #     """Log average training speed of entire training process.

    #     Args:
    #         runner (Runner): The runner of the training process.
    #     """
    #     message_hub = runner.message_hub
    #     avg_speed = message_hub.get_scalar('train/speed').mean()
    #     runner.logger.info('Average training speed of entire training process'
    #                        f'is {avg_speed} s/iter')

    def after_train_epoch(self, runner) -> None:
        """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.by_epoch:
            return
        
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        # print(model.epoch)
        model.epoch += 1

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs which start at ``self.save_begin``
        # 2. reach the last epoch of training
        if self.every_n_epochs(runner, self.interval, self.save_begin) or (
                self.save_last and self.is_last_train_epoch(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.epoch + 1} epochs')
            self._save_checkpoint(runner)
