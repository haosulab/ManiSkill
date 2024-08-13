from collections import defaultdict
import gymnasium
import numpy as np
import torch

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
    is_cpu_sim = isinstance(eval_envs, gymnasium.vector.AsyncVectorEnv)
    with torch.no_grad():
        result = defaultdict(list)
        # reset with reconfigure=True to ensure sufficient randomness in object geometries otherwise they will be fixed in GPU sim.
        obs, info = eval_envs.reset(options=dict(reconfigure=True))
        eps_rets = np.zeros(eval_envs.num_envs)
        eps_lens = np.zeros(eval_envs.num_envs)
        eps_success_once = np.zeros(eval_envs.num_envs)
        eps_count = 0
        while eps_count < n:
            if is_cpu_sim:
                action_seq = agent.get_eval_action(torch.Tensor(obs).to(device)).cpu().numpy()
                for i in range(action_seq.shape[1]):
                    eps_lens += 1
                    obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                    eps_rets += rew
                    if truncated.any():
                        break
                    eps_success_once += info["success"]
            else:
                action_seq = agent.get_eval_action(obs)
                for i in range(action_seq.shape[1]):
                    eps_lens += 1
                    obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                    eps_rets += rew.cpu().numpy()
                    if truncated.any():
                        break
                    eps_success_once += info["success"].cpu().numpy()

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if is_cpu_sim:
                    eps_success = []
                    for i in range(eval_envs.num_envs):
                        eps_success.append(info["final_info"][i]['success'])
                else:
                    eps_success = info["final_info"]['success'].cpu().numpy()
                eps_success_once += eps_success
                result['success_once'].append(eps_success_once > 0)
                result['success_at_end'].append(eps_success)
                result['episode_len'].append(eps_lens)
                result['return'].append(eps_rets)
                eps_count += eval_envs.num_envs
                eps_rets = np.zeros(eval_envs.num_envs)
                eps_lens = np.zeros(eval_envs.num_envs)
                eps_success_once = np.zeros(eval_envs.num_envs)
    agent.train()
    result = {k: np.concatenate(v) for k, v in result.items()}
    return result
