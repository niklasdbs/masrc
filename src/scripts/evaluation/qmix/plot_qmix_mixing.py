import glob
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt, cm
from omegaconf import OmegaConf

import main
from datasets.datasets import DataSplit
from trainers import CLDETrainer
from utils.logging.logger import Logger

metric_to_compare = "validation_advanced_metrics/violation_catched_quota"

run = f"multirun/*/*/+experiment=qmix/qmix,n_steps=5,number_of_agents=2/"
#run = f"multirun/*/*/+experiment=qmix/qmix,n_steps=5,number_of_agents=8/"

runs = glob.glob(run)

if len(runs)>1:
    print("Attention multiple runs found, take latest")

run = sorted(runs)[-1]
file_path = Path(run)
with open(file_path / "scalars.json", mode="r") as f:
    lines = f.readlines()
    json_lines = [json.loads(line) for line in lines]
    best_scalars = [(json_line["max_step"], json_line[metric_to_compare]) for json_line in json_lines if
                    metric_to_compare in json_line.keys()]
    best_scalars.sort(key=lambda x: x[1], reverse=True)
    best_step, best_scalar = best_scalars[0]

#best_step = 24800
#best_step = 6000

cfg = OmegaConf.load(file_path / ".hydra/config.yaml")
cfg["mac_output_probs"] = False
cfg["start_learning"] = 0
cfg["action_selector"] = "EpsilonGreedyActionSelector"
main._set_seeds(cfg.seed)

output_loggers = []

writer = Logger(output_loggers)
with torch.no_grad():
    event_log, graph, shortest_path_lookup = main.load_data_for_env(cfg)
    test_env = main._initialize_environment(DataSplit.TEST, event_log, graph, shortest_path_lookup, cfg)
    trainer = CLDETrainer(train_env=test_env, validation_env=test_env, writer=writer, config=cfg)
    policy_agent = trainer.policy_agent
    policy_agent.load_model(Path(file_path/"models").absolute(), best_step, 0)
    policy_agent.set_test(True)
    env_reset_result = test_env.reset(reset_days=True, only_do_single_episode=True)
    observations = env_reset_result
    state = test_env.state()
    last_action_for_agents = {agent: -1 for agent in test_env.possible_agents}
    current_episode_transitions = []
    device = policy_agent.device

    number_of_agents = trainer.number_of_agents
    number_of_actions = trainer.number_of_actions

    for step in range(999999999):
        inp = [obs["observation"] for agent, obs in observations.items()]
        inp = np.stack(inp, axis=0)
        inp = torch.from_numpy(inp).to(device)

        action_mask = np.ones((number_of_agents, number_of_actions))
        agents_that_need_to_act = []
        for agent_id, obs in observations.items():
            if obs["needs_to_act"] == 1:
                agents_that_need_to_act.append(agent_id)
            else:
                action_mask[agent_id, :] = 0
                action_mask[
                    agent_id, last_action_for_agents[agent_id]] = 1  # agent can only continue with the current action

        if len(agents_that_need_to_act) >1 and step>200:
            action_mask = torch.from_numpy(action_mask).float().to(device)

            q_values = policy_agent.mac.forward(inp)
            q_values[action_mask == 0] =-np.inf
            q_values = q_values.unsqueeze(0).permute(2,0,1)
            q_tot = policy_agent.mixer.forward(q_values, torch.from_numpy(state).to(device).unsqueeze(0).unsqueeze(0))

            #    agent_to_change = 1
            for agent_to_change in test_env.agents:
                lin_space_steps = 1000
                max_q = q_values.max(dim=0)[0][0]
                q_variable = torch.linspace(q_values[:,:,agent_to_change].min().item(),
                                            q_values[:,:,agent_to_change].max().item(),
                                            steps=lin_space_steps).to(device)
                q_real_q = q_values[:, 0, agent_to_change]
                q_variable = torch.cat([q_variable, q_real_q])

                max_q_before = max_q[:agent_to_change]
                max_q_after = max_q[agent_to_change+1:]

                var_mesh_list = []
                if len(max_q_before) > 0:
                    var_mesh_list.extend(max_q_before)

                var_mesh_list.append(q_variable)

                if len(max_q_after) > 0:
                    var_mesh_list.extend(max_q_after)

                var_mesh = torch.meshgrid(*var_mesh_list, indexing="xy")
                q_tot_var = policy_agent.mixer.forward(torch.dstack(var_mesh).flatten(2), torch.from_numpy(state).to(device).unsqueeze(0).unsqueeze(0))

                q_tot_var = q_tot_var.detach().cpu().numpy().flatten()
                q_variable = q_variable.detach().cpu().numpy()
                fig, ax = plt.subplots()
                ax.scatter(q_variable[:lin_space_steps], q_tot_var[:lin_space_steps], s=0.1, alpha=0.3)
                ax.scatter(q_real_q.detach().cpu().numpy(), q_tot_var[lin_space_steps:], s=1.0, color="red")
                ax.set_xlabel(f"Q_{agent_to_change}")
                ax.set_ylabel("Q_tot")
                plt.show()

            if number_of_agents == 2:
                q_values = q_values.squeeze(1)
                x, y = torch.meshgrid(q_values[ :, 0], q_values[:, 1], indexing="ij")
                z = policy_agent.mixer.forward(torch.dstack([x, y]),
                                                  torch.from_numpy(state).to(device).unsqueeze(0).unsqueeze(0))
                z = z.squeeze(-1)
                q_values_0 = q_values[:, 0].detach().cpu().numpy()
                q_values_1 = q_values[:, 1].detach().cpu().numpy()
                q_tot = q_tot.squeeze(1).squeeze(1).detach().cpu().numpy()
                # X = np.arange(q_values_0.min(), q_values_0.max(), 0.001)
                # Y = np.arange(q_values_1.min(), q_values_1.max(), 0.001)
                # X, Y = np.meshgrid(X, Y)
                # Z = policy_agent.mixer.forward(torch.from_numpy(np.stack([X, Y])).permute(1, 2, 0).to(device).float(),
                #                                   torch.from_numpy(state).to(device).unsqueeze(0).unsqueeze(0)).squeeze(-1)
                # Z = Z.detach().cpu().numpy()
                ax = plt.figure().add_subplot(projection='3d')
                # ax.plot_surface(X,Y,Z, cmap=cm.coolwarm, alpha=0.6)
                ax.plot_surface(x.detach().cpu().numpy(),
                                y.detach().cpu().numpy(),
                                z.detach().cpu().numpy(),
                                cmap=cm.coolwarm,
                                alpha=0.6)
                ax.scatter(q_values[ :, 0].max().item(),
                           q_values[ :, 1].max().item(),
                           z[q_values[ :, 0].argmax(), q_values[:, 1].argmax()].item(),
                           color="yellow")
                ax.scatter(x.detach().cpu().numpy(),
                           y.detach().cpu().numpy(),
                           z.detach().cpu().numpy(),
                           alpha=0.1,
                           s=0.1)
                # ax.scatter(q_values_0,q_values_1,q_tot, color="green")
                plt.show()

        actions, _ = policy_agent.act(observations, global_state=state)
        for agent, action in actions.items():
            last_action_for_agents[agent] = action

        next_observations, rewards, discounted_rewards, dones, infos = test_env.step(actions)

        done = any(dones.values())
        state = test_env.state()
        observations = next_observations

        if done:
            break
