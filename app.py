"""
Telecom Egypt Intelligent Assistant
واجهة Streamlit الرئيسية
"""

import streamlit as st
from rag_pipeline import TelecomRAG
from document_processor import process_uploaded_file
import os

# إعدادات الصفحة
st.set_page_config(
    page_title="WE Assistant - Telecom Egypt",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للشكل
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #5E2CA5, #2D1B4E);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-weight: bold;
    }

    .stChatMessage {
        background-color: #F5F3FA;
        border-radius: 12px;
        padding: 12px;
        border-left: 5px solid #5E2CA5;
    }

    .source-link {
        font-size: 0.85rem;
        color: #5E2CA5;
        font-weight: bold;
        text-decoration: none;
    }

    .stButton>button {
        background-color: #5E2CA5;
        color: white;
        border-radius: 8px;
    }

    .stButton>button:hover {
        background-color: #FFC20E;
        color: black;
    }
    .main-header h1 {
            color: white;
            }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown("""
<div class="main-header">
    <h1>📞 WE Intelligent Assistant</h1>
    <p>خدمة عملاء ذكية - مدعومة بالذكاء الاصطناعي</p>
</div>
""", unsafe_allow_html=True)


# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = TelecomRAG(model_name="llama3.2:3b")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# ==================== Sidebar ====================
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # اختيار النموذج
    model_option = st.selectbox(
        "اختر نموذج اللغة",
        ["llama3.2:3b", "llama3.2:1b", "mistral"],
        help="النماذج الأكبر تعطي نتائج أفضل لكن أبطأ"
    )
    
    if model_option != st.session_state.rag.llm.model:
        st.session_state.rag = TelecomRAG(model_name=model_option)
        st.session_state.initialized = False
        st.rerun()
    
    st.divider()
    
    # تحميل البيانات
    st.header("🌐 بيانات الموقع الرسمي")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 تحميل البيانات", use_container_width=True):
            with st.spinner("جاري تحميل بيانات الموقع..."):
                documents = st.session_state.rag.load_documents_from_folder()
                if documents:
                    success = st.session_state.rag.create_vectorstore(documents)
                    if success:
                        st.session_state.initialized = True
                        st.success(f"✅ تم تحميل {len(documents)} صفحة بنجاح!")
                    else:
                        st.error("❌ فشل في إنشاء قاعدة البيانات")
                else:
                    st.warning("⚠️ مفيش بيانات موجودة. اعمل تشغيل scraper.py الأول")
    
    with col2:
        if st.button("🔄 إعادة تحميل", use_container_width=True):
            st.session_state.initialized = st.session_state.rag.load_existing_vectorstore()
            if st.session_state.initialized:
                st.success("✅ تم تحميل قاعدة البيانات الموجودة")
            else:
                st.warning("⚠️ مفيش قاعدة بيانات موجودة")
    
    st.divider()
    
    # رفع الملفات
    st.header("📁 رفع مستندات")
    uploaded_files = st.file_uploader(
        "ارفع ملفاتك (PDF, DOCX, TXT, صورة, HTML)",
        type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'html', 'htm'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:

            if file.size > 20 * 1024 * 1024:
                st.warning(f"⚠️ الملف {file.name} كبير جدًا وقد يؤثر على الأداء")
            
            if st.button(f"📄 معالجة {file.name}", key=file.name):
                with st.spinner(f"جاري معالجة {file.name}..."):
                    text = process_uploaded_file(file)

                    
                    if text and not text.startswith("خطأ"):
                        if st.session_state.rag.add_document(text, file.name):
                            st.success(f"✅ تمت إضافة {file.name}")
                        else:
                            st.error(f"❌ فشل في إضافة {file.name}")
                    else:
                        st.error(f"❌ {text}")
    
    st.divider()
    
    # الإحصائيات
    stats = st.session_state.rag.get_stats()
    st.header("📊 الإحصائيات")
    st.metric("عدد القطع النصية", stats.get('chunks_count', 0))
    st.metric("حالة النظام", "جاهز" if st.session_state.initialized else "في انتظار التحميل")
    
    st.divider()
    
    # زر مسح المحادثة
    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # رابط الموقع الرسمي
    st.divider()
    st.markdown("""
    <div style="text-align: center">
        <small>
            🔗 <a href="https://te.eg" target="_blank">الموقع الرسمي</a><br>
            📞 خدمة العملاء: 19777
        </small>
    </div>
    """, unsafe_allow_html=True)

# ==================== Main Chat Area ====================

# عرض المحادثة
chat_container = st.container()

with chat_container:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("📚 المصادر"):
                    for source in message["sources"]:
                        if source.startswith("http"):
                            st.markdown(f"- [{source}]({source})")
                             

# مدخل السؤال
if prompt := st.chat_input("اسأل عن الباقات، الشحن، الدعم الفني، أو أي حاجة تخص WE..."):
    # إضافة سؤال المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # جلب الإجابة
    with st.chat_message("assistant"):
        if not st.session_state.initialized:
            response = "⚠️ لو سمحت الأول اضغط على **'تحميل البيانات'** في الشريط الجانبي عشان نبدأ."
            sources = []
        else:
            with st.spinner("🤔 بفكر في إجابتك..."):
                response, sources = st.session_state.rag.query(prompt)
        
        st.markdown(response)
        
        if sources:
            with st.expander("📚 المصادر"):
                for source in sources:
                    if source.startswith("http"):
                        st.markdown(f"- 🔗 {source}")
                    
    
    # إضافة رد المساعد
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources if 'sources' in dir() else []
    })

# Footer
st.markdown("---")
st.caption("© 2026 Telecom Egypt Intelligent Assistant")