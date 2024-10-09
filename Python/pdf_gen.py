import os
import fitz
import PySimpleGUI as sg

layout = [
    [sg.Text('Select a PDF file:'), sg.Input(key='file'), sg.FileBrowse()],
    [sg.Button('Generate PDF without images')]
]
window = sg.Window('PDF Image Remover', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == 'Generate PDF without images':
        pdf_file = values['file']
        doc = fitz.open(pdf_file)
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            imgblocks = [b for b in blocks if b["type"] == 1]
            for imgblock in imgblocks:
                xref = imgblock["image"][0]
                if not isinstance(xref, int):
                    raise TypeError(f"xref must be an integer, not {type(xref)}")
                # Pass the xref to delete_image
                page.delete_image(xref) 
        new_pdf_file = os.path.splitext(pdf_file)[0] + '--no images.pdf'
        doc.save(new_pdf_file)
        doc.close()
        sg.popup(f'PDF file without images generated: {new_pdf_file}')
window.close()