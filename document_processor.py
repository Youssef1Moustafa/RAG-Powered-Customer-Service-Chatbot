"""
معالجة الملفات المرفوعة: PDF, DOCX, TXT, Image
"""


import os
from typing import Optional
import tempfile


# معالجة PDF
from pypdf import PdfReader

# معالجة DOCX
from docx import Document

# معالجة الصور
from PIL import Image

try:
    import pytesseract
except:
    pytesseract = None
if pytesseract:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def process_pdf(file_path: str) -> str:
    """استخراج النص من PDF"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted or ""
        if len(text.strip()) < 100:
            return "⚠️ الملف لا يحتوي على نص قابل للاستخراج (قد يكون صورة)"
        return text
    except Exception as e:
        return f"خطأ في قراءة PDF: {e}"

def process_docx(file_path: str) -> str:
    """استخراج النص من DOCX"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"خطأ في قراءة DOCX: {e}"

def process_txt(file_path: str) -> str:
    """استخراج النص من TXT"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # تجربة encoding تاني
        with open(file_path, 'r', encoding='cp1256') as f:
            return f.read()
    except Exception as e:
        return f"خطأ في قراءة TXT: {e}"

def process_image(file_path: str) -> str:
    """استخراج النص من الصور باستخدام OCR"""
    try:
        if pytesseract is None:
            return "⚠️ OCR غير متاح (Tesseract غير مثبت)"
        image = Image.open(file_path)
        # استخدام pytesseract مع دعم اللغة العربية
        text = pytesseract.image_to_string(image, lang='ara+eng')
        return text
    except Exception as e:
        return f"خطأ في معالجة الصورة: {e}"

def process_html(file_path: str) -> str:
    """استخراج النص من HTML"""
    try:
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        # إزالة script و style
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        # تنظيف النص
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        return f"خطأ في قراءة HTML: {e}"

def process_uploaded_file(uploaded_file) -> Optional[str]:
    """
    معالجة الملف المرفوع من Streamlit
    
    Args:
        uploaded_file: الملف من st.file_uploader
    
    Returns:
        النص المستخرج أو None في حالة الخطأ
    """
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()
    
    # حفظ الملف مؤقتًا للمعالجة
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        if file_extension == 'pdf':
            text = process_pdf(tmp_path)
        elif file_extension == 'docx':
            text = process_docx(tmp_path)
        elif file_extension == 'txt':
            text = process_txt(tmp_path)
        elif file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
            text = process_image(tmp_path)
        elif file_extension == 'html' or file_extension == 'htm':
            text = process_html(tmp_path)
        else:
            text = f"نوع الملف {file_extension} غير مدعوم"
        
        # تنظيف الملف المؤقت
        os.unlink(tmp_path)
        
        return text
        
    except Exception as e:
        os.unlink(tmp_path)
        return f"خطأ في المعالجة: {str(e)}"