"""Original PACE runner with serve-safe predictor data and fresh actor fields."""
import torch
from rsl_rl.runners.on_policy_predictor_regression_runner import OnPolicyPredictorRegressionRunner
from pace_learn import learn
from warmstart import initialize_actor
import pace_common as common


class PaceReadyRunner(OnPolicyPredictorRegressionRunner):
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        # Original runner pops class_name fields; snapshot BEFORE construction.
        from copy import deepcopy
        from pathlib import Path
        Path(log_dir).mkdir(parents=True,exist_ok=True)
        torch.save(deepcopy(train_cfg),str(Path(log_dir)/'runner_config.pt'))
        super().__init__(env,train_cfg,log_dir=log_dir,device=device)
        assert env.num_actions == 27 and self.alg.storage.actions.shape[-1] == 27
        self._epoch_cpu = torch.full((self._traj_maxlen,env.num_envs),-1,dtype=torch.long)
        self._valid_cpu = torch.zeros((self._traj_maxlen,env.num_envs),dtype=torch.bool)
        self.predictor_optimizer_steps = 0
        self.predictor_samples_used = 0
        self.predictor_updates_to_env = 0
        if common.CHECK_MODE:
            env.check_reset_isolation()
        initialize_actor(self,env,common.STAND/'model_499.pt',common.STAND/'runner_config.pt')
        self.initial_actor = {k:v.detach().cpu().clone() for k,v in self.alg.policy.actor.state_dict().items()}
        self.initial_predictor = {k:v.detach().cpu().clone() for k,v in self._predictor.state_dict().items()}
        common.RUNNER = self

    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        common.phase('LEARN_ENTER',iterations=num_learning_iterations)
        result=learn(self,num_learning_iterations,init_at_random_ep_len)
        common.phase('LEARN_RETURNED')
        return result

    def _record_ball_positions(self):
        sample = self.env.predictor_sample
        assert sample is not None
        i = self._traj_write_idx
        self._traj_buf_cpu[i].copy_(sample['position'].cpu())
        self._gt_buf_cpu[i].copy_(sample['target'].cpu())
        self._epoch_cpu[i].copy_(sample['epoch'].cpu())
        self._valid_cpu[i].copy_(sample['valid'].cpu())
        self._traj_write_idx = (i+1)%self._traj_maxlen
        self._traj_len = min(self._traj_len+1,self._traj_maxlen)

    def _maybe_predict_and_update_env(self):
        if not self._pred_trained or self._traj_len < self.pred_history_len:
            return
        indices = (torch.arange(-self.pred_history_len,0)+self._traj_write_idx)%self._traj_maxlen
        tags = self._epoch_cpu[indices]
        valid = (tags == self.env.serve_epoch.cpu()[None,:]).all(0)
        x = self._traj_buf_cpu[indices].permute(1,0,2).reshape(self.env.num_envs,-1).to(self.device)
        with torch.no_grad():
            pred = self._predictor(x)
        assert torch.isfinite(pred).all(), 'NONFINITE PREDICTOR OUTPUT'
        pred[~valid.to(self.device)] = 0.
        self.env.update_prediction(pred)
        self.predictor_updates_to_env += int(valid.sum())

    def _train_predictor_offline(self):
        first_update=self.predictor_optimizer_steps == 0
        if first_update:
            common.phase('FIRST_PREDICTOR_UPDATE_BEGIN')
        h,l = self.pred_history_len,self._traj_len
        if l < h+1:
            return None
        ix = (torch.arange(-l,0)+self._traj_write_idx)%self._traj_maxlen
        pos,gt,tags,ok = self._traj_buf_cpu[ix],self._gt_buf_cpu[ix],self._epoch_cpu[ix],self._valid_cpu[ix]
        inputs,labels = [],[]
        for t in range(h,l):
            valid = (tags[t-h:t] == tags[t-1:t]).all(0) & ok[t-1]
            if valid.any():
                x = pos[t-h:t].permute(1,0,2).reshape(self.env.num_envs,-1)
                inputs.append(x[valid]); labels.append(gt[t-1,valid])
        if not inputs:
            return None
        x,y = torch.cat(inputs).to(self.device),torch.cat(labels).to(self.device)
        if self.env.add_noise:
            x = x+.02*torch.randn_like(x)
        self._predictor.train()
        total,count = 0.,0
        self.predictor_samples_used += len(x)
        for _ in range(max(1,self.pred_epochs)):
            for start in range(0,len(x),self.pred_batch_size):
                sl = slice(start,start+self.pred_batch_size)
                loss = torch.nn.functional.mse_loss(self._predictor(x[sl]),y[sl])
                assert torch.isfinite(loss), 'NONFINITE PREDICTOR LOSS'
                self._pred_optim.zero_grad(); loss.backward(); self._pred_optim.step()
                total += float(loss.detach()); count += 1
                self.predictor_optimizer_steps += 1
        self._pred_trained = True
        if first_update:
            common.phase('FIRST_PREDICTOR_UPDATE_END',steps=self.predictor_optimizer_steps)
        return total/max(1,count)

    def _track_physical_metrics(self,infos):
        event = infos['pace_events']
        self._tt_serve_hit_flag |= event['hit']
        self._tt_serve_success_flag |= event['return']
        ids = event['serve_end'].nonzero(as_tuple=False).flatten()
        self._tt_serve_total += len(ids)
        self._tt_hit_total += int(self._tt_serve_hit_flag[ids].sum())
        self._tt_succ_total += int(self._tt_serve_success_flag[ids].sum())
        self._tt_serve_hit_flag[ids] = False
        self._tt_serve_success_flag[ids] = False

    def save(self,path,infos=None):
        super().save(path,infos)
        common.verify()
        self.env.safety.verify()
        common.write_report(__import__('pathlib').Path(self.log_dir)/'pace_ready_progress.json',dict(
            iteration=self.current_learning_iteration,checkpoint=str(path),
            policy_actions=27,predictor_optimizer_steps=self.predictor_optimizer_steps,
            predictor_samples_used=self.predictor_samples_used,
            predictor_updates_to_env=self.predictor_updates_to_env,
            totals=self.env.total,warmstart=self.warmstart_audit))
