from __future__ import print_function


import lasagne
import pickle
import os
import make_net
import sys

def process_param_line(line):

            # Split parameter line on :
            s=str.split(line,':')
            s1=str.strip(s[1],' ;\n')
            # Analyze value
            try:
                # Float or int
                a=float(s1)
                if '.' not in s1:
                    a=int(s1)
            # None, True, False, or list of names.
            except ValueError:
                if (s1=='None'):
                    a=None
                elif (s1=='True'):
                    a=True
                elif (s1=='False'):
                    a=False
                else:
                    if '(' in s1:
                            aa=str.split(str.strip(s1,' ()\n'),',')
                            a=[]
                            for aaa in aa:
                                a.append(float(aaa))
                            a=tuple(a)
                    else:
                        s11=s1.split(',')
                        if (len(s11)==1):
                            a=s1
                        else:
                            a=[]
                            for ss in s11:
                                if (ss != ''):
                                    a.append(ss)


            return(s[0],a)

def process_network_line(line,global_drop):
        # break line on the ; each segment is a parameter for the layer of that line
            sss=str.split(line,';')
            lp={}
            for ss in sss:
                # Split between parameter name and value
                s=str.split(ss,':')
                s1=str.strip(s[1],' \n')
                # Process the parameter value
                # A nonlinearity function
                if ('lasagne' in s1):
                        # if ('rectify' in s1):
                        #     lp['non_linearity']=lasagne.nonlinearities.rectify
                        # elif ('rect_sym' in s1):
                        #     lp['non_linearity']=make_net.rect_sym
                        # elif ('sigmoid' in s1):
                        #     lp['non_linearity']=lasagne.nonlinearities.sigmoid
                        # elif ('tanh' in s1):
                        #     lp['non_linearity']=lasagne.nonlinearities.ScaledTanH(scale_in=.5,scale_out=2.4)
                        # elif ('softmax' in s1):
                        #     lp['non_linearity']=lasagne.nonlinearities.softmax
                        # else:
                        #     lp['non_linearity']=lasagne.nonlinearities.linear
                        print('lasagne')
                else:
                    a=''
                    # A number
                    s1=str.strip(s[1],' \n')
                    try:
                        a=float(s1)
                        if '.' not in s1:
                            a=int(s[1])
                    # A tuple, a list or the original string
                    except ValueError:
                        if '(' in s[1]:
                            aa=str.split(str.strip(s1,' ()\n'),',')
                            a=[]
                            try:
                                int(aa[0])
                                for aaa in aa:
                                    a.append(int(aaa))
                                a=tuple(a)
                            except ValueError:
                                for aaa in aa:
                                    a.append(float(aaa))
                                a=tuple(a)
                        elif '[' in s[1]:
                            aa=str.split(str.strip(s1,' []\n'),',')
                            a=[]
                            for aaa in aa:
                                a.append(aaa)
                        elif (s1=='None'):
                            a=None
                        elif (s1=='True'):
                            a=True
                        elif (s1=='False'):
                            a=False
                        else:
                            a=s1
                    # Add a global drop value to drop layers
                    s0=str.strip(s[0],' ')
                    if (s0=='drop' and global_drop is not None):
                        lp[s0]=global_drop
                    else:
                        lp[s0]=a
            return(lp)



def parse_text_file(net_name,NETPARS,lname='layers', dump=False):


        LAYERS=[]
        if (net_name is not None):
            f=open(net_name+'.txt','r')
            for line in f:
                if (dump):
                     print(line,end="")
                line=str.strip(line,' ')
                ll=str.split(line,'#')
                if (len(ll)>1):
                    line=str.strip(ll[0],' ')
                    if(line==''):
                        continue
                else:
                    line=ll[0]
                if ('name' in line or 'dict' in line):
                    if ('global_drop' in NETPARS):
                        gd=NETPARS['global_drop']
                    else:
                        gd=None
                    lp=process_network_line(line,gd)
                    if ('name' in line):
                        LAYERS.append(lp)
                    else:
                        NETPARS[lp['dict']]=lp
                        del NETPARS[lp['dict']]['dict']
                else:
                    [s,p]=process_param_line(line)
                    NETPARS[s]=p

            f.close()
        # Check if NETPARS is using hinge loss use sigmoid non-linearity on final dense layer
        # Otherwise use softmax



            NETPARS[lname]=LAYERS

def dump_pars(NETPARS):
        import collections
        NETPARS = collections.OrderedDict(sorted(NETPARS.items()))
        for key in NETPARS:
            if (type(NETPARS[key]) is not list):
                print(key+":"+str(NETPARS[key]))
        for key in NETPARS:
            if (type(NETPARS[key]) is list):
                print(key)
                for l in NETPARS[key]:
                    print(l,)
                    #print('\n')