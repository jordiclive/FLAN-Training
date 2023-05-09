# FLAN-Training
A repository to train FLAN T5 and other encoder-decoders with multi-gpus!

`pip install -r requirements.txt`
Alter the config.yaml, and specify the data_path, currently expects train.parquet and val.parquet with "prompt", "text" and "summary" columns.

Run python --configs defaults your_override_config

Where your_override_config is specified in configs/config.yaml. You can also override args in the cmd line.


- Todo: Add Flash attention and multi-node slurm script.

