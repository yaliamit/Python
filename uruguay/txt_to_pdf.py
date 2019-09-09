import os

from fpdf import FPDF
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image, ImageDraw, ImageFont, ImageOps

import pdfCropMargins

def create_image(text,file_name):
    img = Image.new('L', (200,40), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", 24)
    draw.text((0, 0), text,255,font=font)
    ibx=img.getbbox()
    img=img.crop(ibx)
    img=ImageOps.invert(img)
    img.show()
    #img.save(file_name)
    # pdf = FPDF(orientation = 'P', unit = 'pt',format=(200,40))
    # pdf.set_auto_page_break(0)
    # pdf.add_page()
    # #pdf.set_margins(-5, 0, 0)
    # pdf.set_left_margin(-20)
    # pdf.set_font("Arial", size=12)
    # pdf.cell(w=0,txt=text, ln=0, border=0)
    # pdf.output('temp.pdf')
    # images = convert_from_path('temp.pdf')
    print("Hello")

create_image('Shit','out.pdf')
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




