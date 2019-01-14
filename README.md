# GGM
graph generative model for molecule

# Train command
python -u vaetrain.py --item_per_cycle=128 --ncpus=16 --save_dir=save_tpsa/ --key_dir=tpsa_keys/ > output_tpsa

part of output_tpsa would be like this

0	0	0	26.472	26.154	0.084	0.234	51.823

1	0	229	3.454	2.821	0.455	0.178	51.861

2	0	458	2.622	2.071	0.413	0.138	51.802

...

16	0	3664	1.308	0.878	0.307	0.123	51.433

17	0	3893	1.258	0.847	0.301	0.111	50.898

# Test command
