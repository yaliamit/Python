import os

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Welcome to Python!", ln=1, align="C")
pdf.output("simple_demo.pdf")


path="/Users/amit/Desktop/luisa-blocks-real"


def produce_image(r):
    with open (r, "r") as f:
        data=f.readlines()
        print(data)

def im_proc(path,rr):
    for r in rr:
        if 'txt' in r:
            produce_image(path+'/'+r)

def check_if_has_images(path):
    rr=os.listdir(path)
    rr.sort()
    if 'tif' in rr[0] or 'txt' in rr[0]:
        im_proc(path,rr)
    else:
        for r in rr:
            if not r.startswith('.'):
                check_if_has_images(path+'/'+r)


check_if_has_images(path)




