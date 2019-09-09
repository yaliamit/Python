import os

path="/Users/amit/Desktop/luisa-blocks-real"


def produce_image(r):
    with open (r, "r") as f:
        data=f.readlines()
        print(data)

def im_proc(rr):
    for r in rr:
        if 'txt' in r:
            produce_image(r)

def check_if_has_images(path):
    rr=os.listdir(path)
    if 'tif' in rr[0]:
        im_proc(rr)
    else:
        for r in rr:
            if not r.startswith('.'):
                check_if_has_images(path+'/'+r)


check_if_has_images(path)




