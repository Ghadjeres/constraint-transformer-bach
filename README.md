# Transformer with constraints on Bach chorales

## Usage
A conda environment is provided in environment.yml. You can load it with
`conda env create -f environment.yml`. The environment is named `public_transformer_bach` and you can activate it with `conda activate public_transformer_bach`.

Then you can run `python main.py --train --config=transformer_bach/bach_decoder_config.py`.
On the first run, the dataset will be created in `$HOME/Data` and you may need to create this folder.
When prompted for the creation of the index table of the dataset, enter `index`.

After building the dataset (takes around 3 hours) training should start.
Models are saved in the `models/` folder.

You can generate from a trained model with `python main.py --load --config=models/model_id/config.py -o`.
The generations will be placed in the `models/model_id/generations` folder.

You choose to reharmonize different melodies by changing the `melody_constraint` variable at the end of `main.py`. Putting `melody_constraint=None` will generate a chorale from scratch.
