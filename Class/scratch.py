elif 'trans' in l['name']:
            conv=[]
            convp=lasagne.layers.Conv2DLayer(input_la, num_filters=l['num_filters'], filter_size=l['filter_size'],
                                nonlinearity=l['non_linearity'],
                                W=lasagne.init.GlorotUniform())
            convp=extra_pars(convp,l)
            conv.append(convp)
            for lc in l['children']:
                if lc['name']=='transpose':
                    WT=T.transpose(convp.W,(0,1,3,2))
                    convc=lasagne.layers.Conv2DLayer(
                        input_la,  num_filters=l['num_filters'], filter_size=l['filter_size'],
                        nonlinearity=l['non_linearity'],W=WT)
                    convc=extra_pars(convc,l)
                conv.append(convc)
            network[l['name']]=lasagne.layers.ConcatLayer(conv,name=l['name'])
elif 'softmax' in l['name']:
            network[l['name']]=sftlayer(input_la,name=l['name'],)

        elif 'nin' in l['name']:
            network[l['name']]=lasagne.layers.NINLayer(input_la,name=l['name'],num_units=l['num_units'],nonlinearity=l['non_linearity'])
        elif 'divide' in l['name']:
            ins=lasagne.layers.get_output_shape(input_la)
            newdim=l['newdim']
            jump=l['jump']
            olddim=ins[2]
            s0=[]
            for x in np.arange(0,olddim-newdim+1,jump):
                s0.append(slice(x,x+newdim))
            layer_list=[]
            for i in range(len(s0)):
                for j in range(len(s0)):
                    lay1=lasagne.layers.SliceLayer(input_la,s0[i],axis=2)
                    lay1=lasagne.layers.SliceLayer(lay1,s0[j],axis=3)
                    if (len(layer_list)==0):
                        convp=lasagne.layers.Conv2DLayer(lay1, num_filters=l['num_filters'], filter_size=l['filter_size'],
                                nonlinearity=l['non_linearity'],
                                W=lasagne.init.GlorotUniform(),name='conv'+l['name'])
                        convp=extra_pars(convp,l)
                    else:
                        convp=lasagne.layers.Conv2DLayer(
                            lay1,  num_filters=l['num_filters'], filter_size=l['filter_size'],
                            nonlinearity=l['non_linearity'],W=layer_list[0].W, b=layer_list[0].b)
                        convp=extra_pars(convp,l)
                    layer_list.append(convp)
            network[l['name']]=lasagne.layers.ElemwiseMergeLayer(layer_list,T.maximum,name=l['name'])
            #network[l['name']]=lasagne.layers.ConcatLayer(layer_list,name=l['name'])