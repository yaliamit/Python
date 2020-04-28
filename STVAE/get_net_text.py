


def process_network_line(line, global_drop):
    # break line on the ; each segment is a parameter for the layer of that line
    if  len(line)==0 or line[0]=='#':
        return None
    sss = str.split(line, ';')
    lp = {}
    for ss in sss:
        # Split between parameter name and value
        s = str.split(ss, ':')
        #s1 = str.strip(s[1], ' \n')
        # Process the parameter value
        # A nonlinearity function
        a = ''
        # A number
        s1 = str.strip(s[1], ' \n')
        try:
            a = float(s1)
            if '.' not in s1:
                a = int(s[1])
        # A tuple, a list or the original string
        except ValueError:
            if '(' in s[1]:
                aa = str.split(str.strip(s1, ' ()\n'), ',')
                a = []
                try:
                    int(aa[0])
                    for aaa in aa:
                        a.append(int(aaa))
                    a = tuple(a)
                except ValueError:
                    for aaa in aa:
                        a.append(float(aaa))
                    a = tuple(a)
            elif '[' in s[1]:
                aa = str.split(str.strip(s1, ' []\n'), ',')
                a = []
                for aaa in aa:
                    a.append(aaa)
            elif (s1 == 'None'):
                a = None
            elif (s1 == 'True'):
                a = True
            elif (s1 == 'False'):
                a = False
            else:
                a = s1
        # Add a global drop value to drop layers
        s0 = str.strip(s[0], ' ')
        if (s0 == 'drop' and global_drop is not None):
            lp[s0] = global_drop
        else:
            lp[s0] = a
    return (lp)

def get_network(layers,nf=None):


    LP=[]
    for line in layers:
        print(line)
        lp = process_network_line(line, None)
        if lp is not None:
            LP += [lp]
    layers_dict=LP
    if (nf is not None):
        LP[0]['num_filters']=nf
    layer_names_to_indices={}
    for i,ll in enumerate(LP):
        layer_names_to_indices[ll['name']]=i
    lnti=layer_names_to_indices

    return lnti, layers_dict
