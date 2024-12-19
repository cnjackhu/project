# RateRegularizer
for forgetting factor 0,0.5, and 1 i create 3 files run_experiments_ff0.0.py and run_experiments_ff1.py and run_experiments_ff0.5.py. Also the corresponding slurm file slurm0.0.sb, slurm0.5.sb,slurm1.sb. Because i want my request gpu hour not exceed 24 hours, each gpu responding for 1 forgetting factor experiment.

for each forgetting factor, i experiment the
```python
'batch_size_val':[32,64,128,256,512,1024]
```
For the case no reg, i run with slurm_baseline.sb which run sgd_noreg_epoch.py, in this file the only difference with sgd_rate_reg_epoch.py is the
```python
loss = batch_loss  # + inv_rate_with_grads
```
Some plot that i get:
![nll](data/nll.png)
![nll](data/acc.png)
