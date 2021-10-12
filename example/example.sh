#Set the place where result saved
outout_root=example/test_2
fasta_path=$outout_root/example.fasta
mkdir $outout_root

#Download example fasta and save to outout_root
wget "https://symbiosis.iis.sinica.edu.tw/PC_6/data/example.txt" --no-check-certificate -O $fasta_path

#Train model with example fasta
python3 amp_gan/train.py --f $fasta_path -o $outout_root --b 8 --s 10 --e 100
