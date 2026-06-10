"""
RAG Pipeline للـ Telecom Egypt Chatbot - نسخة محدثة
بيستخدم Ollama للنماذج و ChromaDB للتخزين
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from typing import List, Tuple, Dict
import glob
from langdetect import detect
import re
import tempfile 
import shutil


def clean_text(text: str) -> str:
    # إزالة unicode invalid
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    
    # إزالة control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    
    return text.strip()
class TelecomRAG:
    def __init__(self, model_name: str = "llama3-70b-8192"):
        """
        تهيئة نظام RAG
        
        Args:
            model_name: اسم نموذج Ollama (llama3.2:3b أو llama3.2:1b)
        """
        # إعدادات النماذج
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",model_kwargs={'device': 'cpu'},encode_kwargs={'normalize_embeddings': True})
        
        self.llm = ChatGroq(model=model_name,temperature=0.6,api_key=os.getenv("GROQ_API_KEY"))
        
        # مكان حفظ قاعدة البيانات
        self.persist_dir = os.path.join(tempfile.gettempdir(), "data/chroma_db")
        
        # تقسيم النصوص (Chunking)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", "،", " ", ""],
            length_function=len,
        )
        
        # الـ Prompt المصمم خصيصًا
        self.prompt_template = """
أنت موظف خدمة عملاء محترف لشركة WE Egypt.

========================
ROLE
========================

مهمتك هي مساعدة العميل اعتمادًا على:

1- المعلومات الموجودة داخل CONTEXT.
2- المحادثة السابقة CHAT HISTORY.

إذا كان السؤال عام أو اجتماعي (مثل: السلام عليكم، شكراً، صباح الخير، كيف حالك)
فيمكنك الرد بشكل طبيعي دون الحاجة إلى معلومات من CONTEXT.

إذا احتوى السؤال على إساءة أو ألفاظ غير لائقة:

- حافظ على الاحترافية.
- لا ترد بإساءة.
- اطلب من العميل توضيح المشكلة أو السؤال بشكل محترم.
========================
LANGUAGE RULES
========================

- إذا كان العميل يتحدث العربية → رد بالعربية المصرية العامية.
- إذا كان العميل يتحدث الإنجليزية → رد بالإنجليزية.
- لا تخلط بين اللغتين.

========================
CONVERSATION MEMORY
========================

- استخدم المعلومات الموجودة في المحادثة السابقة لفهم السؤال الحالي.
- إذا كان السؤال الحالي مرتبطًا بسؤال سابق، أكمل الحوار بشكل طبيعي.
- لا تطلب من العميل إعادة شرح شيء مذكور بالفعل في المحادثة السابقة.

========================
STRICT RULES
========================

- ممنوع اختراع أي معلومة.
- ممنوع تخمين أسعار أو باقات أو عروض.
- استخدم CHAT HISTORY لفهم الأسئلة المتابعة.
- إذا كان السؤال يعتمد على سؤال سابق، لا تتعامل معه كسؤال جديد.
- استخدم فقط المعلومات الموجودة داخل CONTEXT.
- إذا لم تجد الإجابة داخل CONTEXT قل:

بالعربية:
"مش لاقي المعلومة دي في البيانات المتاحة حاليًا.
ممكن تعيد صياغة السؤال أو توضح تفاصيل أكتر؟"

بالإنجليزية:
"I couldn't find this information in the available data.
Could you rephrase your question or provide more details?"

- لا تذكر أنك نموذج ذكاء اصطناعي.
- لا تذكر كلمة Context أو Documents.
- لا تنشئ روابط غير موجودة.

========================
ANSWER STYLE
========================

- كن ودودًا مثل موظف خدمة العملاء.
- اجعل الإجابة قصيرة ومباشرة.
- إذا كانت الإجابة تحتوي على خطوات استخدم نقاطًا مرقمة.
- إذا كان السؤال عامًا جدًا اطلب تفاصيل إضافية.

أمثلة:

سؤال:
"النت عندي بطيء"

إجابة جيدة:
"ممكن توضحلي أكتر؟
هل المشكلة في الراوتر المنزلي ولا خط الموبايل؟
ومن إمتى بدأت المشكلة؟"

سؤال:
"عايز أعرف تفاصيل الباقة"

إجابة جيدة:
اعرض تفاصيل الباقة الموجودة في البيانات بشكل منظم.
========================
CUSTOMER SERVICE BEHAVIOR
========================

- إذا كان العميل يسأل عن الإنترنت المنزلي، ابحث أولاً داخل البيانات عن:
  * WE Internet
  * Home Internet
  * DSL
  * Fiber
  * Broadband

- إذا كانت المعلومة موجودة بشكل جزئي، استخدمها بدلاً من رفض الإجابة.

- لا تقل "لم أجد المعلومة" إلا إذا لم توجد أي معلومات ذات صلة نهائياً.
========================
CHAT HISTORY
========================

{history}
========================
CONTEXT
========================

{context}

========================
QUESTION
========================

{question}
========================
DECISION PROCESS
========================

قبل الإجابة:

1- افهم السؤال.
2- افحص CHAT HISTORY.
3- ابحث داخل CONTEXT عن معلومات مرتبطة.
4- إذا وجدت معلومات مرتبطة أجب اعتمادًا عليها.
5- إذا لم تجد أي معلومات مرتبطة اطبق رسالة عدم العثور.

لا تتجاهل CHAT HISTORY.

========================
SCOPE RULES
========================

إذا كان السؤال خارج خدمات WE أو غير متعلق بالاتصالات أو الإنترنت أو الخدمات الموجودة في البيانات:

- أخبر العميل بلطف أنك مختص بخدمات WE فقط.
- اطلب منه سؤالاً متعلقًا بخدمات الشركة.

أمثلة:

"من رئيس مصر؟"
→ أنا مختص بخدمات WE فقط، ممكن أساعدك في أي استفسار يخص الإنترنت أو الموبايل أو خدمات WE.

"احسب 2+2"
→ أنا مختص بخدمات WE فقط.
========================
FINAL ANSWER
========================
"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question", "history"]
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
        try:
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
        except Exception as e:
            print(e)
        self.vectorstore = Chroma.from_texts(
            texts=all_chunks,
            embedding=self.embeddings,
            metadatas=all_metadatas,
            persist_directory=self.persist_dir
        )
        
        # بناء سلسلة الـ RAG
        retriever = self.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.4
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
        if os.path.exists(self.persist_dir):
            try:
                self.vectorstore = Chroma(persist_directory=self.persist_dir,embedding_function=self.embeddings)
                retriever = self.vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 5, "score_threshold": 0.4})
                self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,chain_type="stuff",retriever=retriever,chain_type_kwargs={"prompt": self.prompt},return_source_documents=True)
                return True
            except Exception as e:
                print(f"⚠️ خطأ في تحميل قاعدة البيانات: {e}")
                return False
        return False
   
        
    
    def query(self, question: str, chat_history=None) -> Tuple[str, List[str]]:
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
            history_text = ""

            if chat_history:
                for msg in chat_history[-8:]:
                    history_text += f"{msg['role']}: {msg['content']}\n"
            result = self.qa_chain.invoke(
                {
                    "query": f"""
            Previous Conversation:
            {history_text}

             User Language: {lang}

             Current Question:
             {question}
             """
    }
)
    
            response = result['result']

            response = response.strip()
            
            
            # استخراج المصادر
            sources = []
            for doc in result.get('source_documents', []):
                source = doc.metadata.get('source', '')
                title = doc.metadata.get('title', '')
                
                # فلترة اللينكات الغلط
                if "te.eg" in source or "we.com.eg" in source:
                    sources.append(f"{title} - {source}" if title != "No title" else source)
            
            # إزالة التكرار
            sources = list(dict.fromkeys(sources))
            
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
            retriever = self.vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 5,"score_threshold": 0.4})
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
