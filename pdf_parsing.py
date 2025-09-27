# import pymupdf # imports the pymupdf library
# doc = pymupdf.open("Jay Resume 2026-1.pdf")
# text=""
# for page in doc: # iterate the document pages
#   text += page.get_text() 

# token_count = len(text.strip().split())
# print(token_count)# print the text of the page

# text2 = "\n\n".join([page.get_text() for page in doc])
# token_count2 = len(text2.strip().split())
# # print(token_count)


## --------------------------------------------------------------------------------##

from pypdf import PdfReader

reader = PdfReader("Jay Resume 2026-1.pdf")
# text = ""
# for page in reader.pages:
#     text+=page.extract_text() + "\n\n"
    
# token_count=len(text.strip().split())
# print(token_count)


# reader = PdfReader("D:/College_related/Placements Prep/Resources/foundation of LLMs.pdf")
reader = PdfReader("D:/Resources/Resources for CDC.pdf")
text = ""
for page in reader.pages:
    text+=page.extract_text() + "\n\n"
    
token_count=len(text.strip().split())
print(text+'\n\n'+100*'-'+'\n\n', token_count, len(reader.pages))