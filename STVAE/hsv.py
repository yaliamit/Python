import torch


def rgb_to_hsv(input):
    input=input.transpose(1,3)
    sh=input.shape
    input=input.reshape(-1,3)

    mx, inmx = torch.max(input, dim=1)
    mn, inmc = torch.min(input, dim=1)
    df = mx - mn
    h = torch.zeros(input.shape[0],1).to(self.dv)
    ii = [0, 1, 2]
    iid = [[1,2],[2,0],[0,1]]
    shift = [360, 120, 240]

    for i,id,s in zip(ii,iid,shift):
        logi= (df!=0) & (inmx==i)
        h[logi,0] = \
            torch.remainder((60 * (input[logi,id[0]]-input[logi,id[1]])/df[logi] + s),360)

    s = torch.zeros(input.shape[0],1).to(self.)
    s[mx!=0,0]=(df[mx!=0] / mx[mx!=0]) * 100

    v = mx.reshape(input.shape[0],1) * 100

    output = torch.cat((h/360.,s/100.,v/100.),dim=1)

    output = output.reshape(sh).transpose(1,3)
    return output

def hsv_to_rgb(input):
    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)

    hh = input[:,0]
    hh = hh*6
    ihh = torch.floor(hh).type(torch.int32)
    ff = (hh - ihh)[:,None];
    v = input[:,2][:,None]
    s = input[:,1][:,None]
    p = v * (1.0 - s)
    q = v * (1.0 - (s * ff))
    t = v * (1.0 - (s * (1.0 - ff)));

    output=torch.zeros_like(input)
    output[ihh==0,:] = torch.cat((v[ihh==0],t[ihh==0],p[ihh==0]),dim=1)
    output[ihh==1,:] = torch.cat((q[ihh==1],v[ihh==1],p[ihh==1]),dim=1)
    output[ihh==2,:] = torch.cat((p[ihh==2],v[ihh==2],t[ihh==2]),dim=1)
    output[ihh==3,:] = torch.cat((p[ihh==3],q[ihh==3],v[ihh==3]),dim=1)
    output[ihh==4,:] = torch.cat((t[ihh==4],p[ihh==4],v[ihh==4]),dim=1)
    output[ihh==5,:] = torch.cat((v[ihh==5],p[ihh==5],q[ihh==5]),dim=1)

    output=output.reshape(sh)
    output=output.transpose(1,3)
    return output


# PARS = {}
# PARS['data_set'] = 'cifar10'
# PARS['num_train'] = 10
# PARS['nval'] = 0
#
# train, val, test, image_dim = get_data(PARS)
#
# tr=torch.from_numpy(train[0].transpose(0,3,1,2))
#
# otr=rgb_to_hsv(tr)
#
# otr[:,0,:,:]=torch.remainder(otr[:,0,:,:]+.2,1.)
#
# otb=hsv_to_rgb(otr)
#
# print("hello")