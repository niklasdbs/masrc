from typing import Optional

import torch


@torch.no_grad()
def calculate_n_step_targets(dones: torch.Tensor,
                             dts: Optional[torch.Tensor],
                             mask: torch.Tensor,
                             rewards: torch.Tensor,
                             target_max_q_values: torch.Tensor,
                             gamma: float,
                             n_steps: int,
                             semi_markov: bool) -> torch.Tensor:
    assert not semi_markov or dts is not None

    if n_steps == 1:
        # td(1)
        if semi_markov:
            targets = rewards + (1 - dones) * torch.pow(gamma, dts) * target_max_q_values
        else:
            targets = rewards + (1 - dones) * gamma * target_max_q_values
    else:
        # td(n)
        max_seq_length = rewards.size(1)
        n_rewards = torch.zeros_like(rewards)
        if semi_markov:
            max_dts = torch.zeros_like(dts)

            for t in range(max_seq_length):
                n_rewards[:, t, :] = ((mask * rewards)[:, t:t + n_steps, :] * torch.pow(gamma, dts)[:,
                                                                              t:t + n_steps,
                                                                              :]).sum(dim=1)
                max_dts[:, t, :] = torch.sum(dts[:, t:t + n_steps, :], dim=1)
        else:
            gammas = torch.tensor([gamma ** i for i in range(n_steps)],
                                  dtype=torch.float,
                                  device=n_rewards.device)

            gammas = gammas.unsqueeze(0).unsqueeze(-1)

            for t in range(max_seq_length):
                n_rewards[:, t, :] = (
                        (mask * rewards)[:, t:t + n_steps, :] * gammas[:, :(max_seq_length - t), :]).sum(
                        dim=1)

        steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(n_steps).long()

        indices = torch.linspace(0,
                                 max_seq_length - 1,
                                 steps=max_seq_length,
                                 device=steps.device).unsqueeze(1).long()

        n_targets_terminated = torch.gather(target_max_q_values * (1 - dones),
                                            dim=1,
                                            index=steps.long() + indices - 1)
        if semi_markov:
            targets = n_rewards + torch.pow(gamma, max_dts) * n_targets_terminated
        else:
            targets = n_rewards + torch.pow(gamma, steps.float()) * n_targets_terminated
    return targets


# G^{n}_{t} = sum_{l=1}^n gamma^{l-1} * r_{t+l} + gamma^n * f(t+n)
# y^{lambda} = (1-lambda) sum_{n=1}^inf lambda^{n-1}G^{n}_{t}
# use telescoping for efficient calculation


@torch.no_grad()
def build_td_lambda_targets_semi_markov(rewards: torch.Tensor,
                                        dones: torch.Tensor,
                                        mask: torch.Tensor,
                                        max_target_q_values: torch.Tensor,
                                        dts: torch.Tensor,
                                        n_agents: int,
                                        gamma: float,
                                        td_lambda: float) -> torch.Tensor:
    # Assumes max_target_q_values in B*T*N and rewards, dones, mask, dts in (at least) B*T-1*1
    # bootstrap  last  lambda-return  for  not  terminated  episodes
    returns = torch.zeros_like(max_target_q_values)
    returns[:, -1] = max_target_q_values[:, -1] * (1 - dones.sum(dim=1))

    for t in range(returns.shape[1] - 2, -1, -1):  # from (including) T-1 to 0
        returns[:, t] = \
            td_lambda * torch.pow(gamma, dts[:, t]) * returns[:, t + 1] + mask[:, t] \
            * (rewards[:, t] + (1 - td_lambda) * torch.pow(gamma, dts[:, t]) * max_target_q_values[:, t + 1] * (
                    1 - dones[:, t]))


    #assert torch.allclose(returns[:, :-1], calculate_n_step_targets(dones, dts, mask, rewards, max_target_q_values[:,:-1], gamma, 1, True))
    # lambda-return from 0 to T-1, i.e. in B*T-1*N
    return returns[:, :-1]


@torch.no_grad()
def build_td_lambda_targets(rewards: torch.Tensor,
                            dones: torch.Tensor,
                            mask: torch.Tensor,
                            max_target_q_values: torch.Tensor,
                            n_agents: int,
                            gamma: float,
                            td_lambda: float) -> torch.Tensor:
    # Assumes max_target_q_values in B*T*N and rewards, dones, mask in (at least) B*T-1*1
    # bootstrap  last  lambda-return  for  not  terminated  episodes
    returns = torch.zeros_like(max_target_q_values)
    returns[:, -1] = max_target_q_values[:, -1] * (1 - dones.sum(dim=1))

    for t in range(returns.shape[1] - 2, -1, -1):  # from (including) T-1 to 0
        returns[:, t] = td_lambda * gamma * returns[:, t + 1] + mask[:, t] \
                        * (rewards[:, t] + (1 - td_lambda) * gamma * max_target_q_values[:, t + 1] * (1 - dones[:, t]))
    # lambda-return from 0 to T-1, i.e. in B*T-1*N
    return returns[:, :-1]
