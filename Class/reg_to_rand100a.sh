#python runfas.py net=_BP_CIFAR100/igor2Ra_s eta_init=.001 eta_current=.001 igor2Ra_s_OUT
#python runfas.py net=_BP_CIFAR100/igor2Ra_s use_existing=1 start=1 mult=2 eta_init=.0001 eta_current=.0001 igor2Ra_s_1_OUT
ssh yaliamit@linux2.cs.uchicago.edu "sed 's/convR4/convRS4/g' Desktop/Dropbox/Python/Class/_BP_CIFAR100/igor2Ra_s_1.txt > Desktop/Dropbox/Python/Class/_BP_CIFAR100/junk"
ssh yaliamit@linux2.cs.uchicago.edu "mv Desktop/Dropbox/Python/Class/_BP_CIFAR100/junk Desktop/Dropbox/Python/Class/_BP_CIFAR100/igor2Ra_s_1.txt"
python runfas.py net=_BP_CIFAR100/igor2Ra_s mult=2 start=1 use_existing=True write_sparse=True train=False igor2Ra_s_1_OUT
python runfas.py net=_BP_CIFAR100/spigor2Ra_s start=2 mult=3 use_existing=True write_sparse=False train=False spigor2a_s_2_testOUT
python runfas.py net=_BP_CIFAR100/spigor2Ra_s_2 mult=1 start=1 use_existing=1 write_sparse=False mod_net=trymod_randRa100 spigor2Ra_s_2_OUT
#python runfas.py net=_BP_CIFAR100/igor2Ra_s_1 mult=1 start=0 use_existing=True write_sparse=True train=False igor2Ra_s_1_0_1_OUT
#python runfas.py net=_BP_CIFAR100/spigor2Ra_s_1 start=1 mult=2 use_existing=True write_sparse=False eta_current=.001 eta_init=.001 num_epochs=200 spigor2Ra_s_1_1_OUT
