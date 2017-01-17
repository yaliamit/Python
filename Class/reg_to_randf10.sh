python runfas.py net=_BP_CIFAR10/igor2f_s eta_init=.001 eta_current=.001 igor2f_s_OUT
#python runfas.py net=_BP_CIFAR10/igor2f_s use_existing=1 start=1 mult=2 eta_init=.0001 eta_current=.0001 igor2f_s_2_OUT
#python runfas.py net=_BP_CIFAR10/igor2f_s_1 mult=1 start=1 use_existing=1 mod_net=trymod_randf igor2f_s_1_OUT
#python runrcc.py net=_BP_CIFAR10/igor2f_s_2 mult=1 start=0 use_existing=1 igor2f_s_2_1_OUT
#python runrcc.py net=_BP_CIFAR10/igor2f_s_2 mult=2 start=1 use_existing=1 igor2f_s_2_2_OUT
ssh yaliamit@linux2.cs.uchicago.edu "sed 's/convR4/conv4RS/g' Desktop/Dropbox/Python/Class/_BP_CIFAR10/igor2f_s_1.txt > Desktop/Dropbox/Python/Class/_BP_CIFAR10/junk"
ssh yaliamit@linux2.cs.uchicago.edu "cp Desktop/Dropbox/Python/Class/_BP_CIFAR10/junk Desktop/Dropbox/Python/Class/_BP_CIFAR10/igor2f_s_1.txt"
python runfas.py net=_BP_CIFAR10/igor2f_s mult=2 start=1 use_existing=True write_sparse=True train=False igor2f_s_2_OUT
python runfas.py net=_BP_CIFAR10/spigor2f_s start=2 mult=3 use_existing=True write_sparse=False train=False spigor2f_s_2_OUT
#python runrcc.py net=_BP_CIFAR10/spigor2f_s_2 start=1 mult=2 use_existing=True write_sparse=False eta_current=.1 eta_init=.1 num_epochs=200 spigor2f_s_2_1_OUT
#python runrcc.py net=_BP_CIFAR10/spigor2f_s_2 start=2 mult=3 num_epochs=100 use_existing=True write_sparse=False eta_current=.1 eta_init=.1 spigor2f_s_3_1_OUT
