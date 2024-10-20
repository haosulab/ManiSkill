from collections import defaultdict
import gymnasium
import numpy as np
import torch

from mani_skill.utils import common

def update_track_info(infos, ts_tracker, all_time_actions=None):
    if "final_info" in infos: # infos is a dict
        indices = np.where(infos["_final_info"])[0] # not all envs are done at the same time
        for i in indices:
            ts_tracker[i] = 0
            if all_time_actions != None:
                all_time_actions[i] = 0
    return ts_tracker, all_time_actions

def evaluate(n: int, agent, eval_envs, eval_kwargs):
    stats, num_queries, temporal_agg, max_timesteps, device, sim_backend = eval_kwargs.values()

    # determine if visual obs (rgb or rgbd) is used
    if "example_visual_obs" in stats:
        use_visual_obs = True
        #use_depth = stats["example_visual_obs"].shape[1] > 3 # (num_cams, C, 224, 224)
        pre_process = lambda s_obs: (s_obs - stats['state_mean'].cuda()) / stats['state_std'].cuda()
    else:
        use_visual_obs = False
        pre_process = lambda s_obs: (s_obs - stats['state_mean'].cuda()) / stats['state_std'].cuda()
    post_process = lambda a: a * stats['action_std'].cuda() + stats['action_mean'].cuda()

    # create action table for temporal ensembling
    action_dim = stats['action_mean'].shape[-1]
    query_frequency = num_queries
    if temporal_agg:
        query_frequency = 1
        num_queries = num_queries
        with torch.no_grad():
            all_time_actions = torch.zeros([eval_envs.num_envs, max_timesteps, max_timesteps+num_queries, action_dim]).cuda()
    else:
        actions_to_take = torch.zeros([eval_envs.num_envs, query_frequency, action_dim]).cuda()
        all_time_actions = None
    # tracks timestep for each environment.
    ts_tracker = {key: 0 for key in range(eval_envs.num_envs)}

    agent.eval()
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            # pre-process obs
            if use_visual_obs:
                obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
                obs['state'] = pre_process(obs['state'])  # (num_envs, obs_dim)
            else:
                obs = common.to_tensor(obs, device)
                obs = pre_process(obs)  # (num_envs, obs_dim)

            # query policy
            # TODO: query only when ts_tracker[i] % query_frequency == 0
            action_seq = agent.get_action(obs)  # (num_envs, num_queries, action_dim)

            # compute action to take at the current timestep
            raw_action_stacked = []
            if temporal_agg:
                # temporal ensemble
                for env_idx in ts_tracker:
                    ts = ts_tracker[env_idx]
                    all_time_actions[env_idx, ts, ts:ts+num_queries] = action_seq[env_idx]  # (num_queries, 8)
                    actions_for_curr_step = all_time_actions[env_idx, :, ts]  # (max_timesteps, 8)
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)  # (max_timesteps)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]  # (num_populated, 8)

                    # raw_action computed for each env as num_populated could vary in each env.
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))  # (num_populated,)
                    exp_weights = exp_weights / exp_weights.sum()  # (num_populated,)
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)  # (num_populated, 1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0)  # (8)
                    raw_action_stacked.append(raw_action)
            else:
                for env_idx in ts_tracker:
                    ts = ts_tracker[env_idx]
                    if ts % query_frequency == 0:
                        actions_to_take[env_idx] = action_seq[env_idx]  # (num_queries, 8)
                    raw_action = actions_to_take[env_idx, ts % query_frequency]  # (8)
                    raw_action_stacked.append(raw_action)
            raw_action = torch.stack(raw_action_stacked)

            # post-process actions
            # TODO: post-processing adds action_mean to zero-delta actions
            action = post_process(raw_action)  # (num_envs, 8)
            if sim_backend == "cpu":
                action = action.cpu().numpy()

            # step the environment
            obs, rew, terminated, truncated, info = eval_envs.step(action)
            for env_idx in ts_tracker: ts_tracker[env_idx] += 1

            # collect episode info
            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs

            # timestep and table should be set to zero if env is done
            ts_tracker, all_time_actions = update_track_info(info, ts_tracker, all_time_actions)

    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
    return eval_metrics
