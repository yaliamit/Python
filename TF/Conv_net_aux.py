import parse_net_pars as pp
import subprocess as commands




def process_parameters(net):
    PARS = {}
    pp.parse_text_file(net, PARS, lname='layers', dump=True)
    PARS['step_size'] = PARS['eta_init']
    Rstep_size = list(PARS['force_global_prob'])[1] * PARS['step_size']
    print('Rstep_size', Rstep_size)
    PARS['Rstep_size'] = Rstep_size
    PARS['nonlin_scale'] = .5

    return PARS

def print_results(type,epoch,lo,ac):
    print("Final results: epoch", str(epoch))
    print(type+" loss:\t\t\t{:.6f}".format(lo))
    print(type+" acc:\t\t\t{:.6f}".format(ac))

def plot_OUTPUT(name='OUTPUT',code='',first=None,last=None):

    import numpy as np
    import pylab as py
    py.ion()
    havetrain=False
    oo=commands.check_output('grep Posi ' + name + '.txt  | cut -d" " -f2,3', shell=True)
    bp=[]
    bt=np.fromstring(commands.check_output('grep Train ' + name + '.txt | grep acc | cut -d":" -f2',shell=True),sep='\n\t\t\t')
    loss=np.fromstring(commands.check_output('grep Train ' + name + '.txt | grep loss | cut -d":" -f2',shell=True),sep='\n\t\t\t')
    fig=py.figure(2)
    py.plot(loss)
    py.figure(1)
    bv=np.fromstring(commands.check_output('grep Val ' + name + '.txt | grep acc | cut -d":" -f2',shell=True),sep='\n\t\t\t')

    ss='grep aggegate ' + name + '.txt | cut -d"," -f4 | cut -d")" -f1'
    try:
        aa=commands.check_output(ss,shell=True)
        atest=np.fromstring(aa,sep='\n\t\t\t')
        print(atest)
        if (type(atest) is np.ndarray and len(atest) > 0):
            atest = atest[-1]
        # ss = 'grep Post-train ' + name + '.txt | grep acc | cut -d":" -f2'
        # atrain = np.fromstring(commands.check_output(ss, shell=True), sep='\n\t\t\t')
        # if (type(atrain) is np.ndarray and len(atrain) > 0):
        #     havetrain = True
        #     atrain = atrain[-1]
    except:
        print('aggeg not found')


    print('Final',atest) #,atrain)
    if (first is not None and last is not None):
        bt=bt[first:last]
        bv=bv[first:last]
        if (bp!=[]):
            bp=bp[first:last]
        print(bv[-1],bt[-1])
    else:
        print(len(bt),bv[-1],bt[-1])
        if (havetrain>0):
            py.plot(len(bt)-2, atest, 'go', markersize=4)
            #py.plot(len(bt)-2, atrain, 'bo', markersize=4)
    py.plot(bt,label='train '+code)
    py.plot(bv,label='val '+code)
    if (bp!=[]):
        py.plot(bp,label='Pos')
    py.legend(loc=4)

    py.show()