# AMP-GAN
The GAN model to generate AMPs/ AVPs and other peptides.

## Installation
To clone the package and install dependency packages:
```
git clone https://github.com/lsbnb/amp_gan.git
cd amp_gan
pip3 install -r requirements.txt
```

## How to use
1. Call the library:
```python3
#The path of input file
from amp_gan.train import main as train_gan
fasta_path=""
#The folder to save result
outout_root=""
train_gan(fasta_path,outout_root,batch_size=8,step=10,epoch=100)
```

2. Call the script:
```shell
python3 amp_gan/train.py --f $fasta_path -o $outout_root --b 8 --s 10 --e 100
```

## To get more detail of the usage
The detail settings for using train.py would be shown by using the follwoing codes
```shell
python3 amp_gan/train.py -h
```

## Examples
There are two example to be used.
1.  Jupyter-notebook example is located in example/example.ipynb.
2.  Script example is located in example/example.sh.

## Reference
If you find AMP-GAN useful, please consider citing: [Intelligent De Novo Design of Novel Antimicrobial Peptides against Antibiotic-Resistant Bacteria Strains, IJMS, 2023](https://www.mdpi.com/1422-0067/24/7/6788)  
```

