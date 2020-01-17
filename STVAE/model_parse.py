#import parse_net_pars as pp

def process_param_line(line):
    # Split parameter line on :
    s = str.split(line, ':')
    s1 = str.strip(s[1], ' ;\n')
    # Analyze value
    try:
        # Float or int
        a = float(s1)
        if '.' not in s1:
            a = int(s1)
    # None, True, False, or list of names.
    except ValueError:
        if (s1 == 'None'):
            a = None
        elif (s1 == 'True'):
            a = True
        elif (s1 == 'False'):
            a = False
        else:
            if '(' in s1:
                aa = str.split(str.strip(s1, ' ()\n'), ',')
                a = []
                for aaa in aa:
                    a.append(float(aaa))
                a = tuple(a)
            else:
                s11 = s1.split(',')
                if (len(s11) == 1):
                    a = s1
                else:
                    a = []
                    for ss in s11:
                        try:
                            aa = int(ss)
                            a.append(aa)
                        except ValueError:
                            if (ss != ''):
                                a.append(ss)

    return (s[0], a)




def parse_text_file(net_name, NETPARS, lname='layers', dump=False):
    LAYERS = []
    if (net_name is not None):
        f = open(net_name + '.txt', 'r')
        for line in f:
            if (dump):
                print(line)
            line = str.strip(line, ' ')
            ll = str.split(line, '#')
            if (len(ll) > 1):
                line = str.strip(ll[0], ' ')
                if (line == ''):
                    continue
            else:
                line = ll[0]
            if ('name' in line or 'dict' in line):
                if ('global_drop' in NETPARS):
                    gd = NETPARS['global_drop']
                else:
                    gd = None
                lp = process_network_line(line, gd)
                if ('name' in line):
                    LAYERS.append(lp)
                else:
                    NETPARS[lp['dict']] = lp
                    del NETPARS[lp['dict']]['dict']
            else:
                [s, p] = process_param_line(line)
                NETPARS[s] = p

        f.close()
        # Check if NETPARS is using hinge loss use sigmoid non-linearity on final dense layer
        # Otherwise use softmax

        NETPARS[lname] = LAYERS


def dump_pars(NETPARS):
    import collections
    NETPARS = collections.OrderedDict(sorted(NETPARS.items()))
    for key in NETPARS:
        if (type(NETPARS[key]) is not list):
            print(key + ":" + str(NETPARS[key]))
    for key in NETPARS:
        if (type(NETPARS[key]) is list):
            print(key)
            for l in NETPARS[key]:
                if (key == 'layers'):
                    print(l, )
                else:
                    print(l, " ")

def process_parameters(net):
    PARS = {}
    parse_text_file(net, PARS, lname='layers', dump=True)

    return PARS




