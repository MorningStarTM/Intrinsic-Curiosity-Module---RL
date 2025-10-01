# Graph-based Intrinsic Curiosity Module (ICM)

This project extends the Intrinsic Curiosity Module (ICM) idea into the power grid control setting using graph neural networks.



## Environment Setup
- python == 3.10.13
- grid2op == 1.10.5
- lightsim2grid == 0.7.5


## Create Conda Environment
```bash
conda env create -f environment.yml
conda activate ICM
```

## Scripts for Train
### MLP based Actor Critic
```bash
python main.py actor_critic
```

### GAT based Actor Critic
```bash
python main.py gat_actor_critic

```

### ICM
```bash
python main.py icm
```