# Run full network with softmax
#python runfas.py net=_BP_CIFAR100/igor2g_s num_epochs=300 igor2g_s_OUT
#python runfas.py net=_BP_CIFAR100/igor2g_s_1 use_existing=1 start=1 mult=1 mod_net=trymod_randg num_epochs=100 igor2g_s_1_0a_OUT
#mv _cs/Amodels/igor2g_s_1_0.npy _rcc/Amodels/igor2g_s_1_0a.npy
#mv _cs/Amodels/igor2g_s_1_0.txt _rcc/Amodels/igor2g_s_1_0a.txt
# Convert the last conv layer to sparse layer.
# Run original network replacing top 2 dense layers with R layers.
#python runfas.py net=_BP_CIFAR100/igor2g_s_1 mult=1 start=1 use_existing=1 mod_net=trymod_randga write_sparse=False num_epochs=300 igor2g_s_1_0_OUT
#ssh yaliamit@linux2.cs.uchicago.edu "sed 's/conv4/conv4S/g' Desktop/Dropbox/Python/Class/_BP_CIFAR100/igor2g_s_1_0.txt > Desktop/Dropbox/Python/Class/_BP_CIFAR100/junk"
#ssh yaliamit@linux2.cs.uchicago.edu "cp Desktop/Dropbox/Python/Class/_BP_CIFAR100/junk Desktop/Dropbox/Python/Class/_BP_CIFAR100/igor2g_s_1_0.txt"
#python runfas.py net=_BP_CIFAR100/igor2g_s_1 mult=1 start=0 use_existing=True write_sparse=True train=False igor2g_s_1_1_OUT
#python runfas.py net=_BP_CIFAR100/spigor2g_s_1 start=1 mult=2 use_existing=True write_sparse=False train=False spigor2g_s_1_2_testOUT
# Remove top 2 layers and replace with R layers update top 3 layers, sparse + 2 Rdense.
#python runfas.py net=_BP_CIFAR100/spigor2g_s_1 mult=2 start=1 use_existing=1 write_sparse=False num_epochs=100 spigor2g_s_1_3_OUT
python runfas.py net=_BP_CIFAR100/spigor2g_s_1 mult=3 start=2 use_existing=1 write_sparse=False num_epochs=100 spigor2g_s_1_4_OUT
