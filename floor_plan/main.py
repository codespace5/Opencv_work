import fitz  # PyMuPDF

# Specify the PDF file path
pdf_file = "1.pdf"

# Open the PDF file
pdf_document = fitz.open(pdf_file)

# Iterate through the pages in the PDF
for page_number in range(pdf_document.page_count):
    page = pdf_document.load_page(page_number)

    # Get the images on the page
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_data = base_image["image"]

        # Save the image to a file
        image_filename = f"page_{page_number + 1}_image_{img_index + 1}.png"
        with open(image_filename, "wb") as image_file:
            image_file.write(image_data)

# Close the PDF document
pdf_document.close()