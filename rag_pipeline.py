"""
RAG Pipeline للـ Telecom Egypt Chatbot - نسخة محدثة
بيستخدم Ollama للنماذج و ChromaDB للتخزين
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from typing import List, Tuple, Dict
import glob
from langdetect import detect
import re

def clean_text(text: str) -> str:
    # إزالة unicode invalid
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    
    # إزالة control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    
    return text.strip()
class TelecomRAG:
    def __init__(self, model_name: str = "mistral"):
        """
        تهيئة نظام RAG
        
        Args:
            model_name: اسم نموذج Ollama (llama3.2:3b أو llama3.2:1b)
        """
        # إعدادات النماذج
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.3,
            base_url="http://localhost:11434"
        )
        
        # مكان حفظ قاعدة البيانات
        self.persist_dir = "data/chroma_db"
        
        # تقسيم النصوص (Chunking)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", "،", " ", ""],
            length_function=len,
        )
        
        # الـ Prompt المصمم خصيصًا
        self.prompt_template = """
أنت مساعد خدمة عملاء لشركة WE في مصر.

🎯 مهم جدًا:

- استخدم نفس لغة المستخدم بدقة:
    - لو السؤال بالعربي → رد باللهجة المصرية العامية فقط
    - لو السؤال بالإنجليزي → رد بالإنجليزي فقط
- ممنوع خلط اللغات في نفس الرد

- خلي الكلام بسيط وطبيعي زي موظف خدمة عملاء
- متستخدمش لغة رسمية تقيلة

⚠️ قواعد صارمة:
- استخدم فقط المعلومات الموجودة في السياق
- ممنوع تخمين أو اختراع أي معلومة
- لو المعلومة غير موجودة → قول:
  "مش متأكد من المعلومة دي من البيانات المتاحة"
- لو مفيش لينك واضح في البيانات → متكتبش أي لينك
- متكتبش [Link] أو أي placeholder

📚 السياق:
{context}

❓ السؤال:
{question}

💬 الإجابة:
- نفس لغة السؤال 100%
- قصيرة وواضحة
- استخدم نقط لو فيه خطوات
- أسلوب friendly زي خدمة العملاء
"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        self.vectorstore = None
        self.qa_chain = None
    
    def load_documents_from_folder(self, folder_path: str = "data/website_pages") -> List[Dict]:
        """تحميل النصوص من الملفات"""
        documents = []
        
        if not os.path.exists(folder_path):
            print(f"❌ المجلد {folder_path} غير موجود!")
            return documents
        
        txt_files = glob.glob(f"{folder_path}/*.txt")
        txt_files = [f for f in txt_files if not f.endswith('_summary.txt')]
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                url = lines[0].replace('URL: ', '') if lines[0].startswith('URL:') else "unknown"
                title = lines[1].replace('Title: ', '') if len(lines) > 1 else "No title"
                
                content_start = 0
                for i, line in enumerate(lines):
                    if line.startswith('==='):
                        content_start = i + 1
                        break
                
                actual_content = '\n'.join(lines[content_start:])
                
                if len(actual_content.strip()) > 50:
                    documents.append({
                        'text': actual_content,
                        'metadata': {
                            'source': url,
                            'title': title,
                            'file': os.path.basename(file_path)
                        }
                    })
                    
            except Exception as e:
                print(f"⚠️ خطأ في قراءة {file_path}: {e}")
        
        print(f"📄 تم تحميل {len(documents)} مستند")
        return documents
    
    def create_vectorstore(self, documents: List[Dict]) -> bool:
        """تحويل النصوص إلى Embeddings وتخزينها"""
        if not documents:
            print("❌ لا توجد مستندات للمعالجة")
            return False
        
        print("🔄 جاري تقسيم النصوص...")
        all_chunks = []
        all_metadatas = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc['text'])
            all_chunks.extend(chunks)
            all_metadatas.extend([doc['metadata']] * len(chunks))
        
        print(f"📊 تم إنشاء {len(all_chunks)} قطعة نصية")
        
        print("🔄 جاري إنشاء embeddings والتخزين...")
        # 🔧 تم التعديل: إزالة persist() لأن Chroma 0.4.x يحفظ تلقائيًا
        self.vectorstore = Chroma.from_texts(
            texts=all_chunks,
            embedding=self.embeddings,
            metadatas=all_metadatas,
            persist_directory=self.persist_dir
        )
        
        # بناء سلسلة الـ RAG
        retriever = self.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)
        
        self.qa_chain = RetrievalQA.from_chain_type(
    llm=self.llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": self.prompt},
    return_source_documents=True
)
        
        print("✅ تم بناء RAG Pipeline بنجاح")
        return True
    
    def load_existing_vectorstore(self) -> bool:
        """تحميل قاعدة البيانات الموجودة"""
        if os.path.exists(self.persist_dir):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
                
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": self.prompt},
                    return_source_documents=True
                )
                print("✅ تم تحميل قاعدة البيانات الموجودة")
                return True
            except Exception as e:
                print(f"⚠️ خطأ في تحميل قاعدة البيانات: {e}")
        
        return False
    
    def query(self, question: str) -> Tuple[str, List[str]]:
        """
        إرسال سؤال إلى البوت
        
        Returns:
            (الإجابة, قائمة المصادر)
        """
        if not self.qa_chain:
            return "⚠️ النظام لسة محملش البيانات. الرجاء تحميل البيانات أولاً من القائمة الجانبية.", []
        
        try:
            # 🔥 detect language
            try:
                lang = detect(question)
            
            except:
                lang = "ar"
            
            # 🔧 تم التعديل: استخدام invoke بدل الاتصال المباشر
            result = self.qa_chain.invoke({"query": f"User language: {lang}\nQuestion: {question}"})
            response = result['result']

            response = response.strip()
            
            
            # استخراج المصادر
            sources = []
            valid_sources = []

            for doc in result.get('source_documents', []):
                 
                 source = doc.metadata.get('source', '')
                 # فلترة اللينكات الغلط
                 if "te.eg" in source or "we.com.eg" in source:
                      
                      valid_sources.append(source)
        

            # إزالة التكرار
            sources = list(set(valid_sources))
            
            return response, sources
            
        except Exception as e:
            return f"❌ حدث خطأ: {str(e)}", []
    
    def add_document(self, text: str, source_name: str) -> bool:

        """إضافة مستند جديد"""
        text = clean_text(text)
        try:
            
            if not text or len(text.strip()) < 50:
                return False
            # 🔥 limit ذكي للنص
            if len(text) > 200000:
                text = text[:200000]
    
        
            chunks = self.text_splitter.split_text(text)
            metadatas = [{'source': source_name, 'is_user_doc': True}] * len(chunks)
        
            if self.vectorstore:
                self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
            
            else:
                self.vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory=self.persist_dir
            )
            
        
            # إعادة بناء الـ chain مع البيانات الجديدة
            retriever = self.vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 6,"fetch_k": 20,"lambda_mult": 0.7})
            self.qa_chain = RetrievalQA.from_chain_type(
    llm=self.llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": self.prompt},
    return_source_documents=True
)
            
        
            return True
        except Exception as e:
            print(f"⚠️ خطأ في إضافة المستند: {e}")
            return False
        
    
    def get_stats(self) -> Dict:
        """إحصائيات عن قاعدة البيانات"""
        if self.vectorstore:
            try:
                count = self.vectorstore._collection.count()
                return {
                    'status': 'loaded',
                    'chunks_count': count,
                    'persist_dir': self.persist_dir
                }
            except:
                pass
        
        return {'status': 'not_loaded', 'chunks_count': 0}