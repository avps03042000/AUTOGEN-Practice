import PyPDF2
import docx
import chardet
import os
def read_file(file_path):
    ext = file_path.lower().split(".")[-1]
    
    try:
        if ext == "pdf":
            text = ""
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()

        elif ext == "docx":
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)

        elif ext == "txt":
            with open(file_path, "rb") as file:
                raw_data = file.read(1024)
            encoding = chardet.detect(raw_data)["encoding"]
            
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()

        else:
            return None

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

file_paths = ["D:\Aditya Program files\AI-AGENT\AUTOGEN\Resume_Agent\Resumes\Sr. React JS Developer resume.pdf", 
              "D:\Aditya Program files\AI-AGENT\AUTOGEN\Resume_Agent\Resumes\Quality Analyst resume.docx", 
              "D:\Aditya Program files\AI-AGENT\AUTOGEN\Resume_Agent\JD\BA.txt"]  

for file in file_paths:
    if os.path.exists(file):
        content = read_file(file)
        print(f"Contents of {file}:\n")
        print(content)
        print("-" * 50) 
    else:
        print(f"File not found: {file}")
