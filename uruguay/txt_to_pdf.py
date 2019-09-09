import os

from fpdf import FPDF
import pdfCropMargins

def create_pdf(text,file_name):
    pdf = FPDF(orientation = 'P', unit = 'pt',format=(200,40))
    pdf.set_auto_page_break(0)
    pdf.add_page()
    pdf.set_margins(0, 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.cell(w=0,txt=text, ln=0, align="L")
    pdf.output('temp.pdf')
    os.system('pdf-crop-margins temp.pdf -o ' +file_name)

create_pdf('Shit','out.pdf')
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




