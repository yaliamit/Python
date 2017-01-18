# Run full network with softmax
python runrcc.py net=_BP_CIFAR10/igor2g_s eta_init=.001 eta_current=.001 igor2g_s_OUT
#python runrcc.py net=_BP_CIFAR10/igor2g_s use_existing=1 start=1 mult=2 eta_init=.0001 eta_current=.0001 igor2g_s_2_OUT
# Remove top 3 layers and replace with R layers for a few steps.
python runrcc.py net=_BP_CIFAR10/igor2g_s_1 mult=1 start=1 use_existing=1 mod_net=trymod_randg num_epochs=1 igor2g_s_1_0_OUT
# Run only newdens layers for speed and not to train convR layer which isn't realistic
python runrcc.py net=_BP_CIFAR10/igor2g_s_1_0 mult=1 start=1 use_existing=1 mod_net=trymod_randga num_epochs=50 igor2g_s_1_0_0_OUT
ssh yaliamit@midway2.rcc.uchicago.edu "sed 's/convR4/conv4RS/g' Desktop/Dropbox/Python/Class/_BP_CIFAR10/igor2g_s_1_0_0.txt > Desktop/Dropbox/Python/Class/_BP_CIFAR10/junk"
ssh yaliamit@midway2.rcc.uchicago.edu "cp Desktop/Dropbox/Python/Class/_BP_CIFAR10/junk Desktop/Dropbox/Python/Class/_BP_CIFAR10/igor2g_s_1_0_0.txt"
python runrcc.py net=_BP_CIFAR10/igor2g_s_0 mult=1 start=0 use_existing=True write_sparse=True train=False igor2g_s_1_0_0_OUT
python runrcc.py net=_BP_CIFAR10/spigor2g_s_0 start=0 mult=1 use_existing=True write_sparse=False train=False spigor2g_s_1_0_0_testOUT
python runrcc.py net=_BP_CIFAR10/spigor2g_s_0 start=0 mult=1 use_existing=True write_sparse=False eta_current=.1 eta_init=.1 num_epochs=200 spigor2g_s_1_0_1.OUT
#python runrcc.py net=_BP_CIFAR10/spigor2g_s_2 start=2 mult=3 num_epochs=100 use_existing=True write_sparse=False eta_current=.1 eta_init=.1 spigor2g_s_3_1_OUT
