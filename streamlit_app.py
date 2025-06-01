import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import re
from sklearn.cluster import KMeans
from pyembroidery import EmbPattern, STITCH, COLOR_CHANGE, END, write_dst
import openai

st.set_page_config(page_title="AI Embroidery Preview", layout="centered")
st.title("🧵 تحويل صورتك إلى تطريز بالخيوط")

st.markdown("""
ارفع صورتك بصيغة PNG شفافة، وهنحوّلها لمعاينة تطريز فورية.
لو حبيت تغيّر الحجم أو الألوان، كلم الذكاء الصناعي اللي تحت.
""")

uploaded_file = st.file_uploader("📤 ارفع الصورة هنا", type=["png"])

num_colors = st.slider("🎨 اختار عدد الألوان للتطريز:", min_value=5, max_value=16, value=8)

image = None
new_image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGBA")
    st.image(image, caption="📌 الصورة الأصلية", use_column_width=True)

    # تبسيط الألوان
    img_np = np.array(image)
    img_flat = img_np.reshape((-1, 4))
    unique_colors = np.unique(img_flat, axis=0)
    st.markdown(f"عدد الألوان الأصلية: {len(unique_colors)}")

    if len(unique_colors) > num_colors:
        st.warning(f"🔁 جاري تبسيط الصورة إلى {num_colors} ألوان...")
        flat_rgb = img_flat[:, :3]
        kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(flat_rgb)
        clustered = kmeans.cluster_centers_.astype(np.uint8)
        labels = kmeans.labels_
        recolored = clustered[labels].reshape(img_np.shape[:2] + (3,))
        new_image = Image.fromarray(recolored.astype(np.uint8))
        st.image(new_image, caption=f"🧶 معاينة الخياطة ({num_colors} ألوان)", use_column_width=True)
    else:
        new_image = image
        st.image(image, caption="🧶 معاينة الخياطة (ألوان أصلية)", use_column_width=True)

    st.success("✔️ جاهز للتعديل أو التحميل - تقدر تستخدم الشات لتعديل الصورة")

    # 🔽 زر لتحويل الصورة إلى ملف DST
    if st.button("🔽 حمّل كملف خياطة DST"):
        pattern = EmbPattern()
        resized = new_image.resize((64, 64))
        img_data = np.array(resized)
        height, width, _ = img_data.shape

        for y in range(height):
            for x in range(width):
                r, g, b = img_data[y, x]
                pattern.add_stitch_absolute(STITCH, x, y)
            pattern.add_command(COLOR_CHANGE)
        pattern.add_command(END)

        with open("output.dst", "wb") as f:
            write_dst(f, pattern)
        with open("output.dst", "rb") as f:
            st.download_button("📥 اضغط لتحميل ملف التطريز DST", f, file_name="embroidery_output.dst")
else:
    st.info("📎 رجاءً ارفع صورة PNG شفافة للبدء.")

# ✅ واجهة محادثة ذكية مع تأثيرات حقيقية على الصورة
st.markdown("---")
st.subheader("🤖 الذكاء الصناعي - اسألني أي تعديل تحبه")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mod_image" not in st.session_state:
    st.session_state.mod_image = new_image

for msg in st.session_state.messages:
    role = "👤 أنت" if msg["role"] == "user" else "🤖 مساعد"
    with st.chat_message(msg["role"]):
        st.markdown(f"**{role}:** {msg['content']}")

prompt = st.chat_input("✍️ اكتب التعديل اللي تحبه...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"**👤 أنت:** {prompt}")

    openai.api_key = st.secrets["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    )
    ai_reply = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(f"**🤖 مساعد:** {ai_reply}")

    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

    # ✅ تنفيذ أوامر تعديل الصورة البسيطة (لو فيه صورة)
    if new_image:
        mod = new_image.copy()
        if "زود الإضاءة" in ai_reply or "زيادة الإضاءة" in ai_reply:
            mod = ImageEnhance.Brightness(mod).enhance(1.5)
        elif "قلل الإضاءة" in ai_reply or "خفض الإضاءة" in ai_reply:
            mod = ImageEnhance.Brightness(mod).enhance(0.7)
        elif "زود التباين" in ai_reply or "زيادة التباين" in ai_reply:
            mod = ImageEnhance.Contrast(mod).enhance(1.5)
        elif "قلل التباين" in ai_reply or "خفض التباين" in ai_reply:
            mod = ImageEnhance.Contrast(mod).enhance(0.7)

        # عرض النتيجة
        st.image(mod, caption="🎨 النتيجة بعد التعديل", use_column_width=True)
        st.session_state.mod_image = mod

    st.success("✔️ جاهز للتعديل أو التحميل - في المرحلة القادمة هنفعل الشات والتصدير لملف DST")
else:
    st.info("📎 رجاءً ارفع صورة PNG شفافة للبدء.")
