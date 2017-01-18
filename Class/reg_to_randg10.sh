# Run full network with softmax
python runrcc.py net=_BP_CIFAR10/igor2g_s eta_init=.001 eta_current=.001 num_epochs=50 igor2g_s_OUT
#python runrcc.py net=_BP_CIFAR10/igor2g_s eta_init=.001 eta_current=.001 num_epochs=50 igor2g_s_OUT
#ssh yaliamit@midway2.rcc.uchicago.edu "sed 's/conv4/conv4S/g' Desktop/Dropbox/Python/Class/_BP_CIFAR10/igor2g_s_1.txt > Desktop/Dropbox/Python/Class/_BP_CIFAR10/junk"
#ssh yaliamit@midway2.rcc.uchicago.edu "cp Desktop/Dropbox/Python/Class/_BP_CIFAR10/junk Desktop/Dropbox/Python/Class/_BP_CIFAR10/igor2g_s_1.txt"
#python runrcc.py net=_BP_CIFAR10/igor2g_s mult=2 start=1 use_existing=True write_sparse=True train=False igor2g_s_2_OUT
#python runrcc.py net=_BP_CIFAR10/spigor2g_s start=2 mult=3 use_existing=True write_sparse=False train=False spigor2g_s_2_testOUT
# Remove top 2 layers and replace with R layers for a few steps.
#python runrcc.py net=_BP_CIFAR10/spigor2g_s_2 mult=1 start=1 use_existing=1 mod_net=trymod_randg write_sparse=False num_epochs=100 spigor2g_s_2_0_OUT
python runrcc.py net=_BP_CIFAR10/igor2g_s_1 mult=1 start=1 use_existing=1 mod_net=trymod_randga10 write_sparse=False num_epochs=100 igor2g_s_1_0_OUT

# Run only newdens layers for speed and not to train convR layer which isn't realistic