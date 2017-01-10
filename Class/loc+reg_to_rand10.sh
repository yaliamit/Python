#python runrcc.py net=_BP_CIFAR10/igor2Q_s eta_init=.001 eta_current=.001 igor2Q_loc_s_OUT
#python runrcc.py net=_BP_CIFAR10/igor2Q_s use_existing=1 start=1 mult=2 eta_init=.0001 eta_current=.0001 igor2Q_loc_s_1_OUT
#python runrcc.py net=_BP_CIFAR10/igor2Q_s mult=2 start=1 use_existing=1 train=False test_loc_OUT
#sed 's/convR3/convRS3/g' _BP_CIFAR10/igor2Q_s_1.txt > _BP_CIFAR10/junk
#mv _BP_CIFAR10/junk _BP_CIFAR10/igor2Q_s_1.txt
#python runrcc.py net=_BP_CIFAR10/igor2Q_s mult=2 start=1 use_existing=True write_sparse=True train=False igor2Q_loc_s_1_OUT
python runrcc.py net=_BP_CIFAR10/spigor2Q_s start=2 mult=3 use_existing=True write_sparse=False train=False sp_loc_testOUT
python runrcc.py net=_BP_CIFAR10/spigor2Q_s_2 mult=1 start=1 use_existing=1 mod_net=trymod_randR10 spigor2Q_loc_s_2_OUT
#python runrcc.py net=_BP_CIFAR10/igor2Q_s_1 mult=1 start=0 use_existing=True write_sparse=True train=False igor2Q_loc_s_1_0_1_OUT
#python runrcc.py net=_BP_CIFAR10/spigor2Q_s_1 start=1 mult=2 use_existing=True write_sparse=False eta_current=.001 eta_init=.001 num_epochs=200 spigor2Q_loc_s_1_1_OUT
