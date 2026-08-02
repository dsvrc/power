# MARL2Grid

MARL2Grid is a benchmark for **multi-agent reinforcement learning (MARL)** in realistic power grid operations capturing the cooperative and decentralized structure of real-world power systems, where multiple operators control different parts of the grid.  

MARL2Grid is built on top of [Grid2Op](https://github.com/rte-france/Grid2Op), a power grid simulation framework developed by RTE France, and provides standardized **multi-agent tasks, action/state spaces, and reward functions** through a [PettingZoo](https://pettingzoo.farama.org/) interface. 

Since the topology optimization (discrete) and redispatching and curtailment (continuous) tasks rely on different Grid2Op versions, we separate the codebases in two repositories. Additional installation, usage, and execution information are available in separate `README.md` files inside the two folders:
    - `Redispatching_Curtailment_Task`
    - `Topology_Task`