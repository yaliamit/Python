import os

path="/Users/amit/Desktop/luisa-blocks-real"
rr=os.listdir(path)

for r in rr:
    if not r.startswith('.'):
        rr_sub=os.listdir(path+'/'+r)
        
        print("hello")




