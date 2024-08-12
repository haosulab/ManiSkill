from collections import defaultdict
import numpy as np
import torch
from mani_skill.utils import common

def collect_episode_info(infos, result):
    if "final_info" in infos: # infos is a dict

        indices = np.where(infos["_final_info"])[0] # not all envs are done at the same time
        for i in indices:
            info = infos["final_info"][i] # info is also a dict
            ep = info['episode']
            result['return'].append(ep['r'][0])
            result['episode_len'].append(ep["l"][0])
            if "success" in info:
                result['success'].append(info['success'])
            if "fail" in info:
                result['fail'].append(info['fail'])
    return result

def evaluate(n, agent, eval_envs, device):
    agent.eval()
    with torch.no_grad():
        result = defaultdict(list)
        # reset with reconfigure=True to ensure sufficient randomness in object geometries otherwise they will be fixed in GPU sim.
        obs, info = eval_envs.reset(options=dict(reconfigure=True))
        eps_rets = np.zeros(eval_envs.num_envs)
        eps_lens = np.zeros(eval_envs.num_envs)
        eps_success = np.zeros(eval_envs.num_envs)
        while len(result['return']) < n:
            action = agent.get_eval_action(torch.Tensor(obs).to(device))
            obs, rew, terminated, truncated, info = eval_envs.step(action.cpu().numpy())
            eps_rets += common.to_numpy(rew)
            eps_lens += 1
            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                import ipdb;ipdb.set_trace()
                eps_lens *= 0
                eps_rets *= 0
                eps_success *= 0
    agent.train()
    return result
