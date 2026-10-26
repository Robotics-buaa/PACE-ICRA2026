# Derived from received PACE predictor runner; original PPO update loop.
import os
import time
import torch

def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
    # initialize writer (copied from parent)
    if self.log_dir is not None and getattr(self, "writer", None) is None and not self.disable_logs:
        # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()
        if self.logger_type == "neptune":
            from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    # Initialize TT metrics at iter 0 so curves appear immediately
    if self.log_dir is not None and not self.disable_logs and getattr(self, "writer", None) is not None:
        try:
            if not hasattr(self, "_tt_logged_init"):
                self.writer.add_scalar("Train/TT_success_rate", 0.0, 0)
                self.writer.add_scalar("Train/TT_hit_rate", 0.0, 0)
                self._tt_logged_init = True
        except Exception:
            pass

    # teacher check
    if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
        raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

    # randomize initial episode lengths (for exploration)
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )

    # start learning
    obs, extras = self.env.get_observations()
    privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
    obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
    self.train_mode()

    # Book keeping (copied from parent)
    from collections import deque as _dq

    ep_infos = []
    rewbuffer = _dq(maxlen=100)
    lenbuffer = _dq(maxlen=100)
    cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    if self.alg.rnd:
        erewbuffer = _dq(maxlen=100)
        irewbuffer = _dq(maxlen=100)
        cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    # Table-tennis serve and hit tracking for training logs
    # Aggregate across rollout steps and log every 100 updates
    self._tt_succ_total: int = 0
    self._tt_hit_total: int = 0
    self._tt_serve_total: int = 0
    try:
        self._tt_serve_success_flag = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self._tt_serve_hit_flag = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
    except Exception:
        self._tt_serve_success_flag = None
        self._tt_serve_hit_flag = None

    # Ensure all parameters are in-synced
    if self.is_distributed:
        print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
        self.alg.broadcast_parameters()

    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations
    for it in range(start_iter, tot_iter):
        start = time.time()
        # Rollout
        with torch.inference_mode():
            for _ in range(self.num_steps_per_env):
                # Sample actions and step
                actions = self.alg.act(obs, privileged_obs)
                obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                # Move to device
                obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                # Same PACE predictor/rollout ordering, but do not swallow exceptions.
                self._record_ball_positions()
                self._maybe_predict_and_update_env()
                obs, refreshed = self.env.refresh_prediction_observations()
                obs = self.obs_normalizer(obs.to(self.device))
                if self.privileged_obs_type is not None:
                    privileged_obs = self.privileged_obs_normalizer(refreshed['observations'][self.privileged_obs_type].to(self.device))
                else:
                    privileged_obs = obs

                self._track_physical_metrics(infos)

                # RL algorithm step processing
                self.alg.process_env_step(rewards, dones, infos)

                # Extract intrinsic rewards (only for logging)
                intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                # Logging bookkeeping mirrors parent
                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    if self.alg.rnd:
                        cur_ereward_sum += rewards
                        cur_ireward_sum += intrinsic_rewards  # type: ignore
                        cur_reward_sum += rewards + intrinsic_rewards
                    else:
                        cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                    if self.alg.rnd:
                        erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cur_ereward_sum[new_ids] = 0
                        cur_ireward_sum[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            # compute returns (same as parent)
            if self.training_type == "rl":
                self.alg.compute_returns(privileged_obs)

        # Train auxiliary predictor on offline trajectory data (optional cutoff)
        pred_loss_val = None
        if self.current_learning_iteration < self.pred_train_until_iters:
            pred_loss_val = self._train_predictor_offline()
        if pred_loss_val is not None:
            self._last_pred_loss = float(pred_loss_val)

        # Update policy (PPO)
        loss_dict = self.alg.update()

        stop = time.time()
        learn_time = stop - start
        self.current_learning_iteration = it

        # Attach predictor loss to logs if available
        if pred_loss_val is not None:
            loss_dict = dict(loss_dict)  # shallow copy for augmentation
            loss_dict["predictor_mse"] = float(pred_loss_val)

        if self.log_dir is not None and not self.disable_logs:
            self.log(locals())
            # Log TT success and hit rates every 100 updates
            if (it + 1) % 100 == 0:
                serve_total = self._tt_serve_total
                succ_rate = (self._tt_succ_total / serve_total) if serve_total > 0 else 0.0
                hit_rate = (self._tt_hit_total / serve_total) if serve_total > 0 else 0.0
                try:
                    self.writer.add_scalar("Train/TT_success_rate", succ_rate, it)
                    self.writer.add_scalar("Train/TT_hit_rate", hit_rate, it)
                except Exception:
                    pass
                # reset counters for next window
                self._tt_succ_total = 0
                self._tt_hit_total = 0
                self._tt_serve_total = 0
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        ep_infos.clear()
        if it == start_iter and not self.disable_logs:
            from rsl_rl.utils import store_code_state

            git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
            if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                for path in git_file_paths:
                    self.writer.save_file(path)

    if self.log_dir is not None and not self.disable_logs:
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    self.learn_completed = True
