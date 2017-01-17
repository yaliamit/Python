python runrcc.py net=_BP_CIFAR10/igor2f_s eta_init=.001 eta_current=.001 igor2f_s_1_OUT
#python runrcc.py net=_BP_CIFAR100/igor2f_s use_existing=1 start=1 mult=2 eta_init=.0001 eta_current=.0001 igor2f_s_2_OUT
python runrcc.py net=_BP_CIFAR100/igor2f_s mult=2 start=1 use_existing=1 train=False testOUT
#python runrcc.py net=_BP_CIFAR100/igor2f_s_2 mult=1 start=1 use_existing=1 mod_net=trymod_randf igor2f_s_2_0_OUT
#python runrcc.py net=_BP_CIFAR100/igor2f_s_2 mult=1 start=0 use_existing=1 igor2f_s_2_1_OUT
#python runrcc.py net=_BP_CIFAR100/igor2f_s_2 mult=2 start=1 use_existing=1 igor2f_s_2_2_OUT
#ssh yaliamit@midway2.rcc.uchicago.edu "sed 's/conv4/conv4S/g' Desktop/Dropbox/Python/Class/_BP_CIFAR100/igor2f_s_2_0.txt > Desktop/Dropbox/Python/Class/_BP_CIFAR100/junk"
#ssh yaliamit@midway2.rcc.uchicago.edu "mv Desktop/Dropbox/Python/Class/_BP_CIFAR100/junk Desktop/Dropbox/Python/Class/_BP_CIFAR100/igor2f_s_2_0.txt"
python runrcc.py net=_BP_CIFAR100/igor2f_s_2 mult=1 start=0 use_existing=True write_sparse=True train=False igor2f_s_2_0_1_OUT
python runrcc.py net=_BP_CIFAR100/spigor2f_s_2 start=1 mult=2 use_existing=True write_sparse=False train=False spf_testOUT
python runrcc.py net=_BP_CIFAR100/spigor2f_s_2 start=1 mult=2 use_existing=True write_sparse=False eta_current=.1 eta_init=.1 num_epochs=200 spigor2f_s_2_1_OUT
#python runrcc.py net=_BP_CIFAR100/spigor2f_s_2 start=2 mult=3 num_epochs=100 use_existing=True write_sparse=False eta_current=.1 eta_init=.1 spigor2f_s_3_1_OUT
