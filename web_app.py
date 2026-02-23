import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from sklearn.cluster import KMeans
from fpdf import FPDF
import base64
import unicodedata
import os
import time
import sqlite3
import hashlib
from datetime import datetime
import warnings
import logging
import glob

# Tắt cảnh báo
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
os.environ["STREAMLIT_SILENCE_WATCHDOG_WARNING"] = "1"
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Hàm dọn dẹp ảnh tạm
def cleanup_old_images(folder=".", prefix="temp_car_", max_age_seconds=300):
    now = time.time()
    for f in glob.glob(f"{prefix}*.jpg"):
        if os.stat(f).st_mtime < now - max_age_seconds:
            try: os.remove(f)
            except: pass

# ==========================================
# 1. CẤU HÌNH & CSS (GIAO DIỆN HIỆN ĐẠI TỰ THÍCH NGHI)
# ==========================================
st.set_page_config(page_title="AutoVision Ultimate", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    /* Card thông số tự đổi màu theo theme sáng/tối */
    .metric-card {
        background-color: rgba(128, 128, 128, 0.08); 
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 15px; 
        padding: 20px; 
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    /* Nút bấm Gradient thương hiệu */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B2B 0%, #FF416C 100%); 
        color: white; 
        border-radius: 10px; 
        font-weight: bold; 
        height: 48px; 
        width: 100%;
        border: none;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4);
    }
    /* Giá tiền cực đại và sắc nét */
    .big-price {
        font-size: 55px; 
        font-weight: 800; 
        color: #4ade80; 
        margin: 10px 0;
        text-align: center;
    }
    /* Header đăng nhập kiểu hiện đại */
    .login-header {
        text-align: center; 
        background: linear-gradient(90deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 38px; 
        font-weight: 900; 
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DATABASE & AUTH (GIỮ NGUYÊN)
# ==========================================
def init_db():
    conn = sqlite3.connect('autovision.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, car_name TEXT, final_price REAL, timestamp TEXT)')
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ('admin', hashlib.sha256(str.encode('123')).hexdigest(), 'admin'))
        conn.commit()
    except Exception: 
        pass
    conn.close()

init_db()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text: return True
    return False

def add_user(username, password):
    conn = sqlite3.connect('autovision.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, make_hashes(password), 'user'))
        conn.commit(); conn.close()
        return True
    except Exception:
        conn.close(); return False

def login_user(username, password):
    conn = sqlite3.connect('autovision.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    data = c.fetchall()
    conn.close()
    if data and check_hashes(password, data[0][1]):
        return data[0][2]
    return False

def save_history_db(username, car_name, price):
    conn = sqlite3.connect('autovision.db')
    c = conn.cursor()
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history (username, car_name, final_price, timestamp) VALUES (?, ?, ?, ?)", (username, car_name, price, time_now))
    conn.commit(); conn.close()

# ==========================================
# 3. QUẢN LÝ SESSION
# ==========================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user_role' not in st.session_state: st.session_state.user_role = ""
if 'username' not in st.session_state: st.session_state.username = ""

if 'damage_cost' not in st.session_state: st.session_state.damage_cost = 0
if 'damage_list' not in st.session_state: st.session_state.damage_list = []
if 'ai_color' not in st.session_state: st.session_state.ai_color = "Chưa quét"
if 'final_price' not in st.session_state: st.session_state.final_price = 0
if 'pdf_image_path' not in st.session_state: st.session_state.pdf_image_path = None
if 'box_status_val' not in st.session_state: st.session_state.box_status_val = "Không lỗi (Hoàn hảo)"
if 'box_color_val' not in st.session_state: st.session_state.box_color_val = "Trắng"
if 'file_uploader_key' not in st.session_state: st.session_state.file_uploader_key = 0

# ==========================================
# 4. ĐĂNG NHẬP
# ==========================================
if not st.session_state.logged_in:
    st.markdown('<p class="login-header">🏎️ HỆ THỐNG AUTOVISION</p>', unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1.6, 1])
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        tab_log, tab_sign = st.tabs(["🔒 Đăng Nhập", "✉️ Đăng Ký"])
        with tab_log:
            username = st.text_input("Tên đăng nhập")
            password = st.text_input("Mật khẩu", type='password')
            if st.button("Đăng Nhập Ngay"):
                role = login_user(username, password)
                if role:
                    st.session_state.logged_in = True
                    st.session_state.user_role = role
                    st.session_state.username = username
                    st.success("Thành công! Đang vào hệ thống...")
                    st.rerun()
                else: st.error("Sai tên đăng nhập hoặc mật khẩu!")
        with tab_sign:
            new_u = st.text_input("Tạo tên User")
            new_p = st.text_input("Tạo Password", type='password')
            if st.button("Đăng Ký Tài Khoản"):
                if add_user(new_u, new_p): st.success("Tạo thành công! Vui lòng đăng nhập.")
                else: st.error("Tên đăng nhập đã tồn tại.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown(f"### 👤 Xin chào, **{st.session_state.username}**!")
    st.markdown("---")
    if st.button("🔄 Làm mới dữ liệu"):
        st.session_state.damage_cost = 0
        st.session_state.damage_list = []
        st.session_state.ai_color = "Chưa quét"
        st.session_state.final_price = 0
        st.session_state.box_status_val = "Không lỗi (Hoàn hảo)"
        st.session_state.box_color_val = "Trắng"
        st.session_state.pdf_image_path = None
        if 'ai_image' in st.session_state: del st.session_state['ai_image']
        st.session_state.file_uploader_key += 1
        st.rerun()

    if st.button("🚪 Đăng Xuất"):
        st.session_state.logged_in = False
        st.session_state.user_role = ""
        st.rerun()

# Quyền ADMIN (Giữ nguyên logic)
if st.session_state.user_role == 'admin':
    st.title("🔑 TRANG QUẢN TRỊ ADMIN")
    conn = sqlite3.connect('autovision.db')
    tab_h, tab_u = st.tabs(["📜 Lịch Sử Định Giá", "👥 Quản Lý Người Dùng"])
    with tab_h:
        try:
            df_hist = pd.read_sql("SELECT * FROM history ORDER BY id DESC", conn)
            st.dataframe(df_hist, use_container_width=True)
            if not df_hist.empty:
                st.write("Biểu đồ giá trị các xe đã định giá:")
                st.bar_chart(df_hist['final_price'])
        except Exception: st.write("Chưa có dữ liệu.")
    with tab_u:
        try:
            df_users = pd.read_sql("SELECT username, role FROM users", conn)
            st.dataframe(df_users, use_container_width=True)
        except Exception: pass
    conn.close()
    st.stop()

# ==========================================
# 6. APP ĐỊNH GIÁ & PDF
# ==========================================
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    s = input_str.replace('đ', 'd').replace('Đ', 'D') 
    nfkd_form = unicodedata.normalize('NFKD', s)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def create_pdf(car_info, final_price, damages, image_path=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, txt="BAO CAO DINH GIA XE (AUTOVISION)", ln=True, align='C')
    pdf.ln(10)
    if image_path and os.path.exists(image_path):
        try:
            pdf.image(image_path, x=50, w=110) 
            pdf.ln(10)
        except: pass
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, txt="1. THONG TIN CHI TIET:", ln=True)
    pdf.set_font("Helvetica", size=11)
    for key, value in car_info.items():
        pdf.cell(0, 8, txt=f"{remove_accents(key)}: {remove_accents(value)}", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, txt="2. TINH TRANG HU HONG:", ln=True)
    if not damages:
        pdf.cell(0, 8, txt="- Xe dep, khong co loi ngoai that.", ln=True)
    else:
        for d in damages: pdf.cell(0, 8, txt=f"- {remove_accents(d)}", ln=True)
    pdf.ln(10)
    pdf.set_draw_color(74, 222, 128); pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 15, txt=f"TONG GIA TRI: {final_price:,.0f} VND", border=1, ln=True, align='C', fill=True)
    return pdf.output()

# Cấu hình AI
@st.cache_data
def load_data():
    try: return pd.read_csv('cardekho.csv')
    except Exception: return pd.DataFrame() 
df = load_data()
car_options = sorted(df['name'].unique().tolist()) if not df.empty else []

@st.cache_resource
def load_ai():
    p_model = None; y_model = None; cols = []
    try: 
        p_model = joblib.load('model_forest.pkl')
        cols = joblib.load('model_columns.pkl')
    except Exception: pass
    try: y_model = YOLO('best.pt')
    except Exception: pass
    return p_model, cols, y_model

price_model, model_cols, damage_model = load_ai()

st.title("🏎️ AUTOVISION ULTIMATE")

tab1, tab2, tab3 = st.tabs(["🔍 ĐỊNH GIÁ & SOI XE", "📊 BÁO CÁO & TRẢ GÓP", "🏆 TOP 10 XE NGON"])

with tab1:
    cleanup_old_images() # Tự động xóa rác
    colL, colR = st.columns([1, 1.3], gap="large")
    with colL:
        st.markdown('<div class="metric-card"><h4>1. Thông Tin Xe</h4>', unsafe_allow_html=True)
        name = st.selectbox("Dòng xe:", car_options)
        st.image(get_brand_logo(name), width=80)
        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Năm SX:", 2000, 2026, 2018)
            km = st.number_input("Odo (Km):", 0, 999999, 50000, step=1000)
            fuel = st.selectbox("Nhiên liệu:", ['Diesel', 'Petrol', 'Electric', 'LPG'])
            owner = st.selectbox("Đời chủ:", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
        with c2:
            trans = st.selectbox("Hộp số:", ['Manual', 'Automatic'])
            seller = st.selectbox("Người bán:", ['Individual', 'Dealer'])
            seats = st.selectbox("Số ghế:", [4, 5, 7, 8], index=1)
            max_power = st.number_input("Mã lực (bhp):", 20.0, 500.0, 80.0)
        plate = st.text_input("💎 Biển số (VD: 51G-999.99):")
        st.markdown('</div>', unsafe_allow_html=True)

    with colR:
        st.markdown('<div class="metric-card"><h4>2. Kiểm Tra Ngoại Thất</h4>', unsafe_allow_html=True)
        img_file = st.file_uploader("Upload ảnh xe:", type=['jpg','png','jpeg'], key=str(st.session_state.file_uploader_key))
        if img_file:
            try:
                img = Image.open(img_file)
                if st.session_state.pdf_image_path and os.path.exists(st.session_state.pdf_image_path):
                    try: os.remove(st.session_state.pdf_image_path)
                    except: pass
                new_base_name = f"temp_car_{int(time.time())}.jpg"
                abs_fixed_path = os.path.abspath(new_base_name)
                img.convert("RGB").save(abs_fixed_path, format="JPEG")
                st.session_state.pdf_image_path = abs_fixed_path
                
                if st.button("🔍 QUÉT AI (Màu & Lỗi)", type="primary"):
                    st.session_state.ai_color = detect_color(img)
                    c_opts = ["Trắng", "Đen", "Bạc/Xám", "Đỏ", "Xanh", "Màu Khác"]
                    st.session_state.box_color_val = st.session_state.ai_color if st.session_state.ai_color in c_opts else "Màu Khác"
                    if damage_model:
                        results = damage_model(img)
                        res_plotted = results[0].plot()
                        st.session_state.ai_image = res_plotted
                        new_ai_name = f"temp_car_ai_{int(time.time())}.jpg"
                        abs_ai_path = os.path.abspath(new_ai_name)
                        Image.fromarray(res_plotted[..., ::-1]).convert('RGB').save(abs_ai_path, format="JPEG")
                        st.session_state.pdf_image_path = abs_ai_path 
                        costs = {'crack': 5000000, 'scratch': 1500000, 'dent': 4000000, 'glass shatter': 8000000, 'lamp broken': 3000000}
                        vn_names = {'crack': 'Nut vo', 'scratch': 'Tray xuoc', 'dent': 'Mop meo', 'glass shatter': 'Be Kinh', 'lamp broken': 'Vo den'}
                        total = 0; d_list = []; detected_classes = []
                        for box in results[0].boxes:
                            cls_name = damage_model.names[int(box.cls[0])]
                            c = costs.get(cls_name, 1000000)
                            total += c
                            d_list.append(f"{vn_names.get(cls_name, cls_name)} (-{c:,.0f} VND)")
                            detected_classes.append(cls_name)
                        st.session_state.damage_cost = total
                        st.session_state.damage_list = d_list
                        if not detected_classes: st.session_state.box_status_val = "Không lỗi (Hoàn hảo)"
                        elif 'glass shatter' in detected_classes or 'lamp broken' in detected_classes: st.session_state.box_status_val = "Bể kính / Vỡ đèn"
                        elif 'crack' in detected_classes or total > 10000000: st.session_state.box_status_val = "Tai nạn nặng"
                        elif 'dent' in detected_classes or total > 4000000: st.session_state.box_status_val = "Móp méo"
                        else: st.session_state.box_status_val = "Trầy xước nhẹ"
            except: st.error("Lỗi file ảnh")

            c1, c2 = st.columns(2)
            with c1: st.image(img, caption=f"Màu AI: {st.session_state.ai_color}", use_container_width=True)
            with c2: 
                if 'ai_image' in st.session_state:
                    st.image(st.session_state.ai_image, caption="AI phát hiện lỗi", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("🛠️ CHỈNH SỬA KẾT QUẢ THỰC TẾ", expanded=True):
            mc1, mc2 = st.columns(2)
            with mc1:
                color_opts = ["Trắng", "Đen", "Bạc/Xám", "Đỏ", "Xanh", "Màu Khác"]
                manual_color = st.selectbox("Màu sắc thực tế:", color_opts, key='box_color_val')
            with mc2:
                dmg_opts = ["Không lỗi (Hoàn hảo)", "Trầy xước nhẹ", "Móp méo", "Bể kính / Vỡ đèn", "Tai nạn nặng"]
                manual_status = st.selectbox("Tình trạng hư hỏng:", dmg_opts, key='box_status_val')

    if st.button("💰 ĐỊNH GIÁ XE", use_container_width=True):
        if price_model:
            damage_prices = {"Không lỗi (Hoàn hảo)": 0, "Trầy xước nhẹ": 2000000, "Móp méo": 5000000, "Bể kính / Vỡ đèn": 8000000, "Tai nạn nặng": 20000000}
            final_dmg_cost = damage_prices.get(manual_status, 0)
            st.session_state.damage_list = [f"{manual_status} (-{final_dmg_cost:,.0f} VND)"] if final_dmg_cost > 0 else []
            input_df = pd.DataFrame([{'year': year, 'km_driven': km, 'fuel': fuel, 'seller_type': seller, 'transmission': trans, 'owner': owner, 'mileage(km/ltr/kg)': 20.0, 'engine': 1248, 'max_power': max_power, 'seats': seats, 'no_year': 2026 - year}])
            input_df = pd.get_dummies(input_df).reindex(columns=model_cols, fill_value=0)
            base_price = price_model.predict(input_df)[0] * 300
            plate_bonus = 15000000 if ("999" in plate or "888" in plate) else (5000000 if ("68" in plate or "86" in plate) else 0)
            color_bonus = 5000000 if manual_color in ["Trắng", "Đen", "Bạc/Xám"] else -3000000
            final_price = base_price - final_dmg_cost + plate_bonus + color_bonus
            st.session_state.final_price = final_price 
            save_history_db(st.session_state.username, name, final_price)
            
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; border: 2px solid #4ade80;">
                <h3 style='margin:0; opacity: 0.8;'>GIÁ THỊ TRƯỜNG ƯỚC TÍNH</h3>
                <h1 class="big-price">{final_price:,.0f} VNĐ</h1>
            </div>
            """, unsafe_allow_html=True)
        else: st.error("Lỗi Model!")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.header("🖨️ Xuất Báo Cáo PDF")
        if st.session_state.final_price > 0:
            st.success("Đã có kết quả định giá!")
            try:
                car_info = {"Xe": name, "Bien So": plate, "Mau": st.session_state.box_color_val, "Nam SX": year}
                pdf_bytes = create_pdf(car_info, st.session_state.final_price, st.session_state.damage_list, st.session_state.pdf_image_path)
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="BaoCao_DinhGia.pdf"><button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">📥 TẢI FILE PDF (XỊN)</button></a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e: st.error(f"Lỗi tạo PDF: {e}")
        else: st.warning("Vui lòng định giá xe ở Tab 1 trước.")
    with c2:
        st.header("🏦 Tính Trả Góp")
        loan = st.number_input("Số tiền vay:", 100000000, 5000000000, 300000000, step=10000000)
        rate = st.number_input("Lãi suất (%/năm):", 5.0, 15.0, 9.0)
        year_loan = st.slider("Vay trong (năm):", 1, 10, 5)
        pay = (loan * rate/100/12) + (loan / (year_loan*12))
        st.info(f"👉 Trả mỗi tháng: {pay:,.0f} VNĐ")

with tab3:
    st.header("🏆 Top 10 Xe Đáng Mua")
    budget = st.number_input("Ngân sách (VNĐ):", 0, 5000000000, 400000000, step=50000000)
    if st.button("Tìm Xe Ngon"):
        if not df.empty:
            df['price_vnd'] = df['selling_price'] * 300
            res = df[df['price_vnd'] <= budget].sort_values(['year', 'price_vnd'], ascending=[False, True]).head(10)
            for _, row in res.iterrows():
                with st.container():
                    c1, c2, c3 = st.columns([1, 3, 2])
                    c1.image(get_brand_logo(row['name']), width=50)
                    c2.write(f"**{row['name']}** ({row['year']})")
                    c3.success(f"{row['price_vnd']:,.0f} VND")
                    st.divider()