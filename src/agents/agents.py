from agents.actor_critic_agent import ActorCriticAgent
from agents.ddqn.ddqn import DDQN
from agents.greedy import Greedy
from agents.joint.joint_ddqn import JointDDQN
from agents.mardam_agent import MARDAM_Agent
from agents.multi.coma import COMA
from agents.multi.qmix_agent import QMIX
from agents.random import RandomAgent
from agents.lerk import Lerk

ddqn = DDQN
greedy = Greedy
random = RandomAgent
mardam = MARDAM_Agent
lerk = Lerk
joint_ddqn = JointDDQN
qmix = QMIX
coma = COMA
ac = ActorCriticAgent