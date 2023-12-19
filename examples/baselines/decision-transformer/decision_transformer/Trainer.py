import time

import numpy as np
import torch


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        dataset,
        batch_size,
        get_batch,
        loss_fn,
        scheduler=None,
        eval_fns=None,
        state_dim=None,
        device=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.state_dim = state_dim
        self.start_time = time.time()
        self.device = device

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs["time/training"] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f"evaluation/{k}"] = v

        logs["time/total"] = time.time() - self.start_time
        logs["time/evaluation"] = time.time() - eval_start
        logs["training/train_loss_mean"] = np.mean(train_losses)
        logs["training/train_loss_std"] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(
            self.dataset, self.batch_size, state_dim=self.state_dim, device=self.device
        )
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            masks=None,
            attention_mask=attention_mask,
            target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds,
            action_preds,
            reward_preds,
            state_target[:, 1:],
            action_target,
            reward_target[:, 1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()


class SequenceTrainer(Trainer):
    def train_step(self):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = self.get_batch(
            self.dataset, self.batch_size, state_dim=self.state_dim, device=self.device
        )
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        loss = self.loss_fn(
            None,
            action_preds,
            None,
            None,
            action_target,
            None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item()
