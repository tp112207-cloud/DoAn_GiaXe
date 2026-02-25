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
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
os.environ["STREAMLIT_SILENCE_WATCHDOG_WARNING"] = "1"
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import glob
import easyocr
import google.generativeai as genai
import re
import admin_pro 
import json
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    # Thêm dòng replace này ngay đầu hàm
    s = input_str.replace('đ', 'd').replace('Đ', 'D') 
    nfkd_form = unicodedata.normalize('NFKD', s)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
def get_brand_logo(car_name):
    if not isinstance(car_name, str): return ""
    brand = car_name.split(' ')[0].lower()
    logos = {
        "maruti": "https://upload.wikimedia.org/wikipedia/en/d/d0/Maruti_Old_Logo.JPG",
        "hyundai": "https://upload.wikimedia.org/wikipedia/commons/4/44/Hyundai_Motor_Company_logo.svg",
        "honda": "https://upload.wikimedia.org/wikipedia/commons/7/7b/Honda_Logo.svg",
        "toyota": "https://upload.wikimedia.org/wikipedia/commons/9/9d/Toyota_carlogo.svg",
        "ford": "https://upload.wikimedia.org/wikipedia/commons/3/3e/Ford_logo_flat.svg",
        "chevrolet": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Chevrolet-logo.png/330px-Chevrolet-logo.png",
        "audi": "https://upload.wikimedia.org/wikipedia/commons/9/92/Audi-Logo_2016.svg",
        "bmw": "https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg",
        "kia": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/KIA_logo3.svg/250px-KIA_logo3.svg.png",
        "mahindra": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Mahindra_logo.svg/500px-Mahindra_logo.svg.png",
        "tata": "https://upload.wikimedia.org/wikipedia/commons/8/8e/Tata_logo.svg"
    }
    return logos.get(brand, "https://cdn-icons-png.flaticon.com/512/741/741407.png")

def detect_color(image):
    try:
        img = np.array(image); img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape
        center_img = img[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        if center_img.size == 0: return "Màu Khác"
        clt = KMeans(n_clusters=1); clt.fit(center_img.reshape((-1, 3)))
        b, g, r = clt.cluster_centers_[0]
        if r>200 and g>200 and b>200: return "Trắng"
        if r<50 and g<50 and b<50: return "Đen"
        if abs(r-g)<20 and r>100: return "Bạc/Xám"
        if r>150 and g<100: return "Đỏ"
        return "Màu Khác"
    except Exception: return "Màu Khác"

def cleanup_old_images(folder=".", prefix="temp_car_", max_age_seconds=300):
    # Tìm các file ảnh tạm có tuổi đời hơn 5 phút (300s)
    now = time.time()
    for f in glob.glob(f"{prefix}*.jpg"):
        if os.stat(f).st_mtime < now - max_age_seconds:
            try: os.remove(f)
            except: pass
# ==========================================
# 1. CẤU HÌNH & CSS (GIỮ NGUYÊN)
# ==========================================
st.set_page_config(page_title="AutoVision Ultimate", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    /* Card thông số: Dùng rgba để tự sáng/tối theo nền */
    .metric-card {
        background-color: rgba(150, 150, 150, 0.1); 
        border: 1px solid rgba(150, 150, 150, 0.2);
        border-radius: 12px; 
        padding: 20px; 
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    /* Nút bấm: Giữ màu Gradient nhưng thêm hiệu ứng hover cho xịn */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B2B 0%, #FF416C 100%); 
        color: white; 
        border-radius: 8px; 
        font-weight: bold; 
        height: 50px; 
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 65, 108, 0.3);
    }
    /* Header: Dùng Gradient text thay vì màu cố định */
    .login-header {
        text-align: center; 
        background: linear-gradient(90deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 35px; 
        font-weight: bold; 
        margin-bottom: 20px;
    }
    .big-price {font-size: 55px; font-weight: 900; color: #4ade80; text-align: center;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DATABASE & AUTH
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
admin_pro.upgrade_database()

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
def save_chat_to_db(chat_id):
    if chat_id in st.session_state.chat_sessions:
        data = st.session_state.chat_sessions[chat_id]
        conn = sqlite3.connect('autovision.db')
        c = conn.cursor()
        msgs_json = json.dumps(data["messages"], ensure_ascii=False)
        # Kiểm tra xem chat này đã có trong DB chưa
        c.execute("SELECT chat_id FROM chat_history WHERE chat_id = ?", (chat_id,))
        if c.fetchone():
            c.execute("UPDATE chat_history SET title=?, messages=?, pinned=?, timestamp=? WHERE chat_id=?", 
                      (data["title"], msgs_json, int(data["pinned"]), data["timestamp"], chat_id))
        else:
            c.execute("INSERT INTO chat_history (chat_id, username, title, messages, pinned, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                      (chat_id, st.session_state.username, data["title"], msgs_json, int(data["pinned"]), data["timestamp"]))
        conn.commit()
        conn.close()

def delete_chat_from_db(chat_id):
    conn = sqlite3.connect('autovision.db')
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()
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
if 'plate_number' not in st.session_state: st.session_state.plate_number = ""
if 'chat_sessions' not in st.session_state:
    # Nếu user đã đăng nhập, ưu tiên lôi lịch sử từ DB ra ngay lập tức
    if st.session_state.logged_in and st.session_state.username and st.session_state.user_role != 'admin':
        conn = sqlite3.connect('autovision.db')
        c = conn.cursor()
        c.execute("SELECT chat_id, title, messages, pinned, timestamp FROM chat_history WHERE username = ? ORDER BY timestamp DESC", (st.session_state.username,))
        rows = c.fetchall()
        conn.close()
        
        if rows: # Nếu tìm thấy lịch sử chat cũ
            loaded_chats = {}
            for r in rows:
                c_id, title, msgs_json, pinned, ts = r
                msgs = json.loads(msgs_json) if msgs_json else []
                loaded_chats[c_id] = {"title": title, "messages": msgs, "pinned": bool(pinned), "timestamp": ts}
            st.session_state.chat_sessions = loaded_chats
            st.session_state.current_chat_id = list(loaded_chats.keys())[0]
        else: # Nếu tài khoản mới tạo, chưa chat bao giờ
            default_id = "chat_" + str(int(time.time()))
            st.session_state.chat_sessions = {default_id: {"title": None, "messages": [], "pinned": False, "timestamp": time.time()}}
            st.session_state.current_chat_id = default_id
    else:
        # Dành cho lúc chưa đăng nhập hoặc là admin
        default_id = "chat_" + str(int(time.time()))
        st.session_state.chat_sessions = {default_id: {"title": None, "messages": [], "pinned": False, "timestamp": time.time()}}
        st.session_state.current_chat_id = default_id
# ==========================================
# 4. ĐĂNG NHẬP
# ==========================================
if not st.session_state.logged_in:
    st.markdown('<p class="login-header">🔐 HỆ THỐNG AUTOVISION - ĐĂNG NHẬP</p>', unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        tab_log, tab_sign = st.tabs(["Đăng Nhập", "Đăng Ký"])
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
                    c_log = sqlite3.connect('autovision.db').cursor()
                    c_log.execute("UPDATE users SET last_login_at = ? WHERE username = ?", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username))
                    c_log.connection.commit()
                    
                    # ---> THÊM ĐOẠN NÀY: Tiêu hủy sổ chat tạm để ép tải lại từ Database <---
                    if 'chat_sessions' in st.session_state: 
                        del st.session_state['chat_sessions']
                    if 'current_chat_id' in st.session_state: 
                        del st.session_state['current_chat_id']
                        
                    st.rerun()
                else: st.error("Sai tên đăng nhập hoặc mật khẩu!")
        with tab_sign:
            new_u = st.text_input("Tạo tên User")
            new_p = st.text_input("Tạo Password", type='password')
            if st.button("Đăng Ký Tài Khoản"):
                if add_user(new_u, new_p): st.success("Tạo thành công! Vui lòng đăng nhập.")
                else: st.error("Tên đăng nhập đã tồn tại.")
    st.stop()

# ==========================================
# 5. ADMIN PANEL & SIDEBAR
# ==========================================
with st.sidebar:
    st.write(f"Xin chào, **{st.session_state.username}**!")
    
    # CHỈ HIỂN THỊ LỊCH SỬ NẾU LÀ USER
    if st.session_state.user_role != 'admin':
        st.markdown("---")
        st.write("💬 **Lịch sử trò chuyện AI**")
        
        if st.button("➕ Đoạn chat mới", use_container_width=True, type="primary"):
            new_id = "chat_" + str(int(time.time()))
            st.session_state.chat_sessions[new_id] = {"title": None, "messages": [], "pinned": False, "timestamp": time.time()}
            st.session_state.current_chat_id = new_id
            save_chat_to_db(new_id) # LƯU VÀO DB
            st.rerun()

        sorted_chats = sorted(
            st.session_state.chat_sessions.items(),
            key=lambda x: (x[1].get('pinned', False), x[1].get('timestamp', 0)),
            reverse=True
        )

        st.markdown("<div style='max-height: 400px; overflow-y: auto; overflow-x: hidden;'>", unsafe_allow_html=True)
        for chat_id, chat_data in sorted_chats:
            if chat_data.get("title"): title = chat_data["title"]
            elif len(chat_data["messages"]) > 0: title = chat_data["messages"][0]['content'][:12] + "..."
            else: title = "Chat mới..."
                
            display_title = ("📌 " if chat_data.get("pinned") else "") + title
            if chat_id == st.session_state.current_chat_id:
                display_title = "👉 " + display_title
                btn_type = "primary"
            else: btn_type = "secondary"
            
            col_chat, col_menu = st.columns([8, 2])
            with col_chat:
                if st.button(display_title, key=f"btn_{chat_id}", use_container_width=True, type=btn_type):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            with col_menu:
                with st.popover("⋮", use_container_width=True):
                    new_name = st.text_input("Đổi tên:", value=title.replace("📌 ", ""), key=f"rn_in_{chat_id}")
                    if st.button("💾 Lưu tên", key=f"rn_btn_{chat_id}", use_container_width=True):
                        st.session_state.chat_sessions[chat_id]["title"] = new_name
                        save_chat_to_db(chat_id) # LƯU ĐỔI TÊN VÀO DB
                        st.rerun()
                        
                    pin_text = "📍 Bỏ ghim" if chat_data.get("pinned") else "📌 Ghim chat"
                    if st.button(pin_text, key=f"pin_btn_{chat_id}", use_container_width=True):
                        st.session_state.chat_sessions[chat_id]["pinned"] = not chat_data.get("pinned", False)
                        save_chat_to_db(chat_id) # LƯU GHIM VÀO DB
                        st.rerun()
                        
                    if st.button("🗑️ Xóa chat", key=f"del_btn_{chat_id}", use_container_width=True):
                        del st.session_state.chat_sessions[chat_id]
                        delete_chat_from_db(chat_id) # XÓA KHỎI DB VĨNH VIỄN
                        if st.session_state.current_chat_id == chat_id:
                            if len(st.session_state.chat_sessions) > 0:
                                st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
                            else:
                                new_id = "chat_" + str(int(time.time()))
                                st.session_state.chat_sessions = {new_id: {"title": None, "messages": [], "pinned": False, "timestamp": time.time()}}
                                st.session_state.current_chat_id = new_id
                                save_chat_to_db(new_id)
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🔄 Làm mới Form định giá"):
            st.session_state.damage_cost = 0
            st.session_state.damage_list = []
            st.session_state.ai_color = "Chưa quét"
            st.session_state.final_price = 0
            st.session_state.box_status_val = "Không lỗi (Hoàn hảo)"
            st.session_state.box_color_val = "Trắng"
            st.session_state.pdf_image_path = None
            st.session_state.plate_number = ""
            if 'ai_image' in st.session_state: del st.session_state['ai_image']
            st.session_state.file_uploader_key += 1
            st.rerun()

    if st.button("🚪 Đăng Xuất"):
        st.session_state.logged_in = False
        st.session_state.user_role = ""
        st.session_state.username = "" # QUAN TRỌNG: Cần xóa Username hiện tại đi
        if 'chat_sessions' in st.session_state: del st.session_state['chat_sessions']
        if 'current_chat_id' in st.session_state: del st.session_state['current_chat_id']
        st.rerun()
if st.session_state.user_role == 'admin':
    # Gọi hàm vẽ giao diện Admin Pro từ file admin_pro.py
    admin_pro.render_admin_dashboard()
    st.stop() # Dừng tại đây, không load giao diện của User ở dưới

# ==========================================
# 6. APP ĐỊNH GIÁ & XỬ LÝ PDF (ĐÃ CẬP NHẬT)
# ==========================================

def create_pdf(car_info, final_price, damages, image_path=None):
    # Cập nhật sử dụng fpdf2 để xuất file ổn định hơn
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, txt="BAO CAO DINH GIA XE (AUTOVISION)", ln=True, align='C')
    pdf.ln(10)
    
    # Chèn ảnh xe an toàn
    if image_path and os.path.exists(image_path):
        try:
            pdf.image(image_path, x=50, w=110) 
            pdf.ln(10)
        except Exception as e:
            pdf.set_font("Helvetica", 'I', 10)
            pdf.cell(0, 10, txt=f"(Khong the hien thi anh: {str(e)})", ln=True, align='C')
    
    # Thông tin xe
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, txt="1. THONG TIN CHI TIET:", ln=True)
    pdf.set_font("Helvetica", size=11)
    for key, value in car_info.items():
        pdf.cell(0, 8, txt=f"{remove_accents(key)}: {remove_accents(value)}", ln=True)
    
    pdf.ln(5)
    
    # Tình trạng hư hỏng
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, txt="2. TINH TRANG HU HONG:", ln=True)
    pdf.set_font("Helvetica", size=11)
    if not damages:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 8, txt="- Xe dep, khong co loi ngoai that.", ln=True)
    else:
        pdf.set_text_color(200, 0, 0)
        for d in damages:
            pdf.cell(0, 8, txt=f"- {remove_accents(d)}", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Giá cuối
    pdf.set_draw_color(74, 222, 128)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 15, txt=f"TONG GIA TRI DINH GIA: {final_price:,.0f} VND", border=1, ln=True, align='C', fill=True)
    
    return pdf.output()

@st.cache_data
def load_data():
    try: return pd.read_csv('cardekho.csv')
    except Exception: return pd.DataFrame() 
df = load_data()

car_options = sorted(df['name'].unique().tolist()) if not df.empty else []

# --- THÊM MODEL OCR ---
@st.cache_resource
def load_ocr():
    try: 
        # Chỉ dùng tiếng Anh ('en') là đủ đọc biển số, tắt GPU để chạy mượt trên server free
        return easyocr.Reader(['en'], gpu=False) 
    except Exception: 
        return None
ocr_reader = load_ocr()

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

tab1, tab2, tab3, tab4 = st.tabs(["🔍 ĐỊNH GIÁ & SOI XE", "📊 BÁO CÁO & TRẢ GÓP", "🏆 TOP 10 XE NGON", "🤖 AI TƯ VẤN"])

with tab1:
    colL, colR = st.columns([1, 1.3], gap="large")
    with colL:
        st.markdown('<div class="metric-card"><h4>1. Thông Tin Xe</h4>', unsafe_allow_html=True)
        name = st.selectbox("Dòng xe:", car_options)
        st.image(get_brand_logo(name), width=80)
        # --- LOGIC AI TỰ ĐỘNG BỐC THÔNG SỐ XE TỪ DATASET ---
        d_seats = 5
        d_engine = 1248.0
        d_power = 80.0
        d_mileage = 20.0
        # Lấy danh sách các số ghế có thể có
        available_seats = [5] # Mặc định
        if not df.empty and name in df['name'].values:
            car_data = df[df['name'] == name]
            if 'seats' in car_data.columns:
                seats_list = car_data['seats'].dropna().unique().astype(int).tolist()
                if seats_list: available_seats = sorted(seats_list)
            
            # Quét thông số engine, power, mileage
            if 'engine' in car_data.columns and not car_data['engine'].isna().all():
                eng_nums = car_data['engine'].astype(str).str.extract(r'(\d+\.?\d*)')[0].dropna().astype(float)
                if not eng_nums.empty: d_engine = float(eng_nums.mode()[0])
            if 'max_power' in car_data.columns and not car_data['max_power'].isna().all():
                pow_nums = car_data['max_power'].astype(str).str.extract(r'(\d+\.?\d*)')[0].dropna().astype(float)
                if not pow_nums.empty: d_power = float(pow_nums.mode()[0])
            if 'mileage' in car_data.columns and not car_data['mileage'].isna().all():
                mil_nums = car_data['mileage'].astype(str).str.extract(r'(\d+\.?\d*)')[0].dropna().astype(float)
                if not mil_nums.empty: d_mileage = float(mil_nums.mode()[0])

        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Năm SX:", 2000, 2026, 2018)
            km = st.number_input("Odo (Km):", 0, 999999, 50000, step=1000)
            fuel = st.selectbox("Nhiên liệu:", ['Diesel', 'Petrol', 'Electric', 'LPG'])
            owner = st.selectbox("Đời chủ:", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
            seats = st.number_input("Số ghế (Auto):", value=d_seats, disabled=True, format="%d")
        with c2:
            trans = st.selectbox("Hộp số:", ['Manual', 'Automatic'])
            seller = st.selectbox("Người bán:", ['Individual', 'Dealer'])
            engine_val = st.number_input("Động cơ CC (Auto):", value=d_engine, disabled=True)
            max_power = st.number_input("Mã lực (Auto):", value=d_power, disabled=True)
            mileage_val = st.number_input("Tiêu hao km/l (Auto):", value=d_mileage, disabled=True)
            
        st.markdown("---")
        plate = st.text_input("💎 Biển số (VD: 51G-999.99):", value=st.session_state.plate_number)
        st.markdown('</div>', unsafe_allow_html=True)
    with colR:
        st.markdown('<div class="metric-card"><h4>2. Kiểm Tra Ngoại Thất</h4>', unsafe_allow_html=True)
        img_file = st.file_uploader("Upload ảnh xe:", type=['jpg','png','jpeg'], key=str(st.session_state.file_uploader_key))
        
        if img_file:
            try:
                img = Image.open(img_file)
                if st.session_state.pdf_image_path and os.path.exists(st.session_state.pdf_image_path):
                    try: os.remove(st.session_state.pdf_image_path)
                    except Exception: pass
                new_base_name = f"temp_car_{int(time.time())}.jpg"
                abs_fixed_path = os.path.abspath(new_base_name)
                img.convert("RGB").save(abs_fixed_path, format="JPEG")
                st.session_state.pdf_image_path = abs_fixed_path
            except Exception:
                st.error("Lỗi file ảnh")
                img = None

            if img:
                if st.button("🔍 QUÉT AI (Màu & Lỗi)", type="primary"):
                    try:
                        st.session_state.ai_color = detect_color(img)
                        c_opts = ["Trắng", "Đen", "Bạc/Xám", "Đỏ", "Xanh", "Màu Khác"]
                        st.session_state.box_color_val = st.session_state.ai_color if st.session_state.ai_color in c_opts else "Màu Khác"
                        
                        if damage_model:
                            results = damage_model(img)
                            res_plotted = results[0].plot()
                            st.session_state.ai_image = res_plotted
                            
                            if st.session_state.pdf_image_path and os.path.exists(st.session_state.pdf_image_path):
                                try: os.remove(st.session_state.pdf_image_path)
                                except Exception: pass
                            
                            new_ai_name = f"temp_car_ai_{int(time.time())}.jpg"
                            abs_ai_path = os.path.abspath(new_ai_name)
                            Image.fromarray(res_plotted[..., ::-1]).convert('RGB').save(abs_ai_path, format="JPEG")
                            st.session_state.pdf_image_path = abs_ai_path 
                            
                            costs = {'crack': 5000000, 'scratch': 1500000, 'dent': 4000000, 'glass shatter': 8000000, 'lamp broken': 3000000}
                            vn_names = {'crack': 'Nứt vỡ', 'scratch': 'Trầy xước', 'dent': 'Móp méo', 'glass shatter': 'Bể Kính', 'lamp broken': 'Vỡ đèn'}
                            
                            total = 0; d_list = []; detected_classes = []
                            for box in results[0].boxes:
                                cls_name = damage_model.names[int(box.cls[0])]
                                c = costs.get(cls_name, 1000000)
                                total += c
                                d_list.append(f"{vn_names.get(cls_name, cls_name)} (-{c:,.0f}VND)")
                                detected_classes.append(cls_name)
                            
                            st.session_state.damage_cost = total
                            st.session_state.damage_list = d_list

                            # --- CẬP NHẬT LOGIC CHẨN ĐOÁN ƯU TIÊN LOẠI LỖI ---
                            if not detected_classes:
                                st.session_state.box_status_val = "Không lỗi (Hoàn hảo)"
                            elif 'glass shatter' in detected_classes or 'lamp broken' in detected_classes:
                                st.session_state.box_status_val = "Bể kính / Vỡ đèn"
                            elif 'crack' in detected_classes or total > 10000000:
                                st.session_state.box_status_val = "Tai nạn nặng"
                            elif 'dent' in detected_classes or total > 4000000:
                                st.session_state.box_status_val = "Móp méo"
                            else:
                                st.session_state.box_status_val = "Trầy xước nhẹ"
                                # --- TÍNH NĂNG MỚI: QUÉT BIỂN SỐ BẰNG EASYOCR (TỪ ẢNH GỐC) ---
                            if ocr_reader:
                                st.toast("🤖 Đang đọc biển số từ ảnh gốc...")
                                try:
                                    # CHUẨN LUÔN: Lấy biến 'img' (ảnh gốc từ file uploader) để đọc, không bị dính chữ của YOLO
                                    img_np = np.array(img) 
                                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                                    
                                    # Tiền xử lý cho rõ chữ: Phóng to x2 và chuyển sang trắng đen
                                    img_cv = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                                    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                                    
                                    # Tiến hành đọc chữ
                                    text_list = ocr_reader.readtext(gray_img, detail=0)
                                    
                                    valid_parts = []
                                    for text in text_list:
                                        # Giữ lại toàn bộ CHỮ CÁI và CON SỐ
                                        clean_text = "".join(e for e in text if e.isalnum()).upper()
                                        if len(clean_text) >= 2:
                                            valid_parts.append(clean_text)
                                            
                                    if valid_parts:
                                        raw_plate = "".join(valid_parts)
                                        
                                        # Cắt bớt nếu nó lỡ đọc luôn số hotline trên tường
                                        if len(raw_plate) > 12:
                                            raw_plate = max(valid_parts, key=len) 
                                            
                                        # BỘ LỌC ĐẶC BIỆT: Ép ký tự thứ 3 thành CHỮ (Sửa lỗi 374 -> 37A)
                                        raw_plate_list = list(raw_plate)
                                        if len(raw_plate_list) >= 5:
                                            fix_dict = {'4': 'A', '8': 'B', '0': 'D', '5': 'S', '2': 'Z', '6': 'G', '7': 'T'}
                                            # Vị trí thứ 3 (index 2) trong biển số VN thường là chữ
                                            if raw_plate_list[2] in fix_dict:
                                                raw_plate_list[2] = fix_dict[raw_plate_list[2]]
                                        raw_plate = "".join(raw_plate_list)
                                        
                                        # TỰ ĐỘNG FORMAT CHÈN DẤU (-) VÀ DẤU (.)
                                        formatted_plate = raw_plate
                                        # Regex bóc tách biển số VN: (2 số đầu + 1 chữ + có thể 1 số) và (4 hoặc 5 số cuối)
                                        match = re.match(r'^(\d{2}[A-Z]\d?)(\d{4,5})$', raw_plate)
                                        if match:
                                            head = match.group(1)
                                            tail = match.group(2)
                                            if len(tail) == 5:
                                                formatted_plate = f"{head}-{tail[:3]}.{tail[3:]}" # VD: 37A-718.60
                                            else:
                                                formatted_plate = f"{head}-{tail}" # VD: 29A-1234
                                                
                                        if len(formatted_plate) >= 4:
                                            st.session_state.plate_number = formatted_plate
                                            st.rerun() # Refresh để tự động điền số
                                        else:
                                            st.toast("⚠️ Tìm thấy chữ nhưng không giống biển số xe lắm!")
                                    else:
                                        st.toast("⚠️ Không thấy biển số trong ảnh gốc!")
                                        
                                except Exception as e:
                                    st.toast(f"Lỗi quét ảnh: {e}")
                        else: st.warning("Chưa có Model AI.")
                    except Exception: pass

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
                st.selectbox("Màu sắc thực tế:", color_opts, key='box_color_val')
            with mc2:
                dmg_opts = ["Không lỗi (Hoàn hảo)", "Trầy xước nhẹ", "Móp méo", "Bể kính / Vỡ đèn", "Tai nạn nặng"]
                st.selectbox("Tình trạng hư hỏng:", dmg_opts, key='box_status_val')

    st.markdown("---")
    if st.button("💰 ĐỊNH GIÁ XE", use_container_width=True):
        # --- KIỂM TRA GIỚI HẠN ĐỊNH GIÁ TỪ ADMIN ---
        conn_limit = sqlite3.connect('autovision.db')
        c_limit = conn_limit.cursor()
        c_limit.execute("SELECT value FROM settings WHERE key = 'valuation_limit'")
        limit_row = c_limit.fetchone()
        val_limit = int(limit_row[0]) if limit_row else 0
        
        can_proceed = True
        if val_limit > 0:
            today_str = datetime.now().strftime("%Y-%m-%d")
            # Đếm xem user này hôm nay đã định giá bao nhiêu lần
            c_limit.execute("SELECT COUNT(*) FROM history WHERE username = ? AND timestamp LIKE ?", (st.session_state.username, f"{today_str}%"))
            today_count = c_limit.fetchone()[0]
            
            if today_count >= val_limit:
                can_proceed = False
                st.error(f"❌ Bạn đã hết lượt định giá hôm nay ({today_count}/{val_limit} lượt). Vui lòng liên hệ Admin hoặc quay lại vào ngày mai!")
        conn_limit.close()
        
        # --- NẾU CHƯA HẾT LƯỢT THÌ MỚI CHO ĐỊNH GIÁ ---
        if can_proceed:
            if price_model:
                damage_prices = {"Không lỗi (Hoàn hảo)": 0, "Trầy xước nhẹ": 2000000, "Móp méo": 5000000, "Bể kính / Vỡ đèn": 8000000, "Tai nạn nặng": 20000000}
                manual_status = st.session_state.box_status_val
                manual_color = st.session_state.box_color_val
                final_dmg_cost = damage_prices.get(manual_status, 0)
                if final_dmg_cost > 0: st.session_state.damage_list = [f"{manual_status} (-{final_dmg_cost:,.0f}VND)"]
                else: st.session_state.damage_list = []
                
                input_df = pd.DataFrame([{
                    'year': year, 'km_driven': km, 'fuel': fuel, 'seller_type': seller,
                    'transmission': trans, 'owner': owner, 
                    'mileage(km/ltr/kg)': mileage_val, 
                    'engine': engine_val,              
                    'max_power': max_power,            
                    'seats': seats,                    
                    'no_year': 2026 - year
                }])
                input_df = pd.get_dummies(input_df).reindex(columns=model_cols, fill_value=0)
                base_price = price_model.predict(input_df)[0] * 300
                
                plate_bonus = 0
                if plate:
                    p = plate.upper().replace(".", "").replace("-", "")
                    if "999" in p or "888" in p: plate_bonus = 15000000
                    elif "68" in p or "86" in p: plate_bonus = 5000000
                
                color_bonus = 5000000 if manual_color in ["Trắng", "Đen", "Bạc/Xám"] else -3000000
                final_price = base_price - final_dmg_cost + plate_bonus + color_bonus
                st.session_state.final_price = final_price 
                
                save_history_db(st.session_state.username, name, final_price)
                st.markdown(f"""
                <div style="background-color:#1f2937; padding:20px; border-radius:15px; text-align:center; border:2px solid #4ade80;">
                    <h3 style='color:#9ca3af; margin:0;'>GIÁ THỊ TRƯỜNG: {base_price:,.0f} VNĐ</h3>
                    <h1 class="big-price">{final_price:,.0f} VNĐ</h1>
                </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.info(f"🎨 Màu {manual_color}: {color_bonus:+,.0f}")
                col2.success(f"💎 Biển số: +{plate_bonus:,.0f}")
                if final_dmg_cost > 0: col3.error(f"📉 {manual_status}: -{final_dmg_cost:,.0f}")
                else: col3.success("✅ Xe đẹp, không trừ tiền")
                # --- TÍNH NĂNG MỚI: BIỂU ĐỒ DỰ BÁO MẤT GIÁ (CẬP NHẬT THEO NĂM THỰC TẾ) ---
                st.markdown("---")
                st.markdown("<h4 style='text-align: center; color: #facc15;'>📉 DỰ BÁO KHẤU HAO GIÁ TRỊ XE TRONG 5 NĂM TỚI</h4>", unsafe_allow_html=True)
                
                # Lấy năm hiện tại tự động (Ví dụ: 2026)
                current_yr = datetime.now().year

                
                predicted_prices = [final_price]
                years_list = [str(current_yr)]
                for y in range(1, 6):
                    # Tạo dữ liệu ảo cho tương lai (Tuổi xe tăng lên, Odo tăng trung bình 15.000km/năm)
                    future_df = pd.DataFrame([{
                        'year': year, 
                        'km_driven': km + (15000 * y), 
                        'fuel': fuel, 'seller_type': seller,
                        'transmission': trans, 'owner': owner, 
                        'mileage(km/ltr/kg)': 20.0,
                        'engine': 1248, 'max_power': max_power, 'seats': seats,
                        'no_year': (current_yr - year) + y 
                    }])
                
                # Format dữ liệu chuẩn với mô hình AI
                future_df = pd.get_dummies(future_df).reindex(columns=model_cols, fill_value=0)
                
                # Dự đoán giá gốc
                future_base_price = price_model.predict(future_df)[0] * 300
                
                # Tính giá cuối cùng (vẫn trừ đi lỗi ngoại thất và cộng biển số/màu sắc ban đầu)
                future_final_price = future_base_price - final_dmg_cost + plate_bonus + color_bonus
                
                # Đảm bảo giá không bị rớt thê thảm xuống số âm
                predicted_prices.append(max(future_final_price, 50000000))

                years_list.append(str(current_yr + y))
                
            # Đóng gói dữ liệu thành bảng để vẽ biểu đồ
            chart_data = pd.DataFrame({
                "Giá trị dự kiến (VNĐ)": predicted_prices
            }, index=years_list)
            
            # Vẽ biểu đồ đường
            st.line_chart(chart_data)
            
            # Nhận xét tự động
            loss_after_5_years = final_price - predicted_prices[-1]
            if loss_after_5_years > 0:
                st.info(f"💡 Dựa trên phân tích AI, ước tính đến năm **{current_yr + 5}** (kèm {15000*5:,} km sử dụng thêm), xe sẽ mất giá khoảng **{loss_after_5_years:,.0f} VNĐ**.")
            else:
                st.info("💡 Xe đang giữ giá rất tốt theo dự báo của AI!")
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
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="BaoCao_DinhGia.pdf"><button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">📥 TẢI FILE PDF (CÓ ẢNH)</button></a>'
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
            for index, row in res.iterrows():
                with st.container():
                    c1, c2, c3 = st.columns([1, 3, 2])
                    c1.image(get_brand_logo(row['name']), width=50) # Hiện logo hãng
                    c2.write(f"**{row['name']}** ({row['year']})")
                    c3.success(f"{row['price_vnd']:,.0f} VND")
                    st.divider()
with tab4:
    st.header("🤖 Cố Vấn AI Chuyên Sâu")
    st.caption("Hỏi bất kỳ điều gì về xe cộ, không cần phải định giá trước!")
    st.markdown("---")
    
    # Lấy key an toàn từ file secrets.toml
    try:
        MY_SECRET_KEY = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        MY_SECRET_KEY = None
        st.error("⚠️ Lỗi hệ thống: Chưa tìm thấy API Key trong file bảo mật .streamlit/secrets.toml")
        
    if MY_SECRET_KEY:
        # --- KIỂM TRA CÔNG TẮC TẮT/BẬT AI TỪ ADMIN ---
        conn_ai = sqlite3.connect('autovision.db')
        c_ai = conn_ai.cursor()
        c_ai.execute("SELECT value FROM settings WHERE key = 'ai_chat_enabled'")
        ai_status = c_ai.fetchone()
        conn_ai.close()
        
        is_ai_enabled = True if (ai_status and ai_status[0] == '1') else False
        
        if not is_ai_enabled:
            st.warning("🚫 Quản trị viên (Admin) đã tạm thời tắt tính năng Trợ lý AI. Vui lòng quay lại sau!")
        else:
            genai.configure(api_key=MY_SECRET_KEY)
            
            # TRỎ VÀO ĐÚNG DỮ LIỆU CỦA ĐOẠN CHAT HIỆN TẠI
            current_chat_data = st.session_state.chat_sessions[st.session_state.current_chat_id]
            current_chat_msgs = current_chat_data["messages"]
                
            # TẠO KHUNG CHAT CỐ ĐỊNH CHIỀU CAO 
            chat_container = st.container(height=450)
            
            with chat_container:
                for msg in current_chat_msgs:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                    
            if prompt := st.chat_input("Ví dụ: Tư vấn giúp tôi mua xe 500 triệu..."):
                current_chat_msgs.append({"role": "user", "content": prompt})
                save_chat_to_db(st.session_state.current_chat_id)
                
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                # XỬ LÝ NGỮ CẢNH
                if st.session_state.final_price > 0:
                    context_str = f"""
                    THÔNG TIN CHIẾC XE NGƯỜI DÙNG VỪA ĐỊNH GIÁ TRÊN HỆ THỐNG:
                    - Dòng xe: {st.session_state.get('name', 'Không rõ')}
                    - Năm sản xuất: {st.session_state.get('year', 'Không rõ')}
                    - Tình trạng hư hỏng: {st.session_state.box_status_val}
                    - Giá hệ thống vừa dự đoán: {st.session_state.final_price:,.0f} VNĐ.
                    """
                else:
                    context_str = "HIỆN TẠI NGƯỜI DÙNG CHƯA ĐỊNH GIÁ CHIẾC XE NÀO."
                    
                system_instruction = f"""
                Bạn là chuyên gia thẩm định và tư vấn xe hơi 15 năm kinh nghiệm.
                {context_str}
                
                Nhiệm vụ bắt buộc:
                1. Trả lời bằng tiếng Việt thân thiện, dễ hiểu.
                2. Nếu người dùng hỏi về chiếc xe vừa định giá, LUÔN TRÌNH BÀY DẠNG BULLET POINT giải thích vì sao có giá đó.
                3. Nếu người dùng hỏi kiến thức xe cộ chung chung, hãy tư vấn nhiệt tình.
                """
                
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Đang xử lí dữ liệu..."):
                            try:
                                model = genai.GenerativeModel("gemini-3-flash-preview")
                                
                                # Lấy đúng lịch sử của đoạn chat này đưa cho AI
                                gemini_history = []
                                for m in current_chat_msgs[:-1]: 
                                    role = "model" if m["role"] == "assistant" else "user"
                                    gemini_history.append({"role": role, "parts": [m["content"]]})
                                    
                                chat = model.start_chat(history=gemini_history)
                                full_prompt = f"[HƯỚNG DẪN DÀNH CHO AI]:\n{system_instruction}\n\n[CÂU HỎI CỦA NGƯỜI DÙNG]:\n{prompt}"
                                response = chat.send_message(full_prompt)
                                
                                st.markdown(response.text)
                                current_chat_msgs.append({"role": "assistant", "content": response.text})
                                save_chat_to_db(st.session_state.current_chat_id)
                                
                                # Cập nhật số lần chat cho Admin
                                try:
                                    db_conn = sqlite3.connect('autovision.db')
                                    db_cursor = db_conn.cursor()
                                    db_cursor.execute("SELECT value FROM settings WHERE key = 'total_ai_chats'")
                                    row = db_cursor.fetchone()
                                    if row:
                                        new_count = str(int(row[0]) + 1)
                                        db_cursor.execute("UPDATE settings SET value = ? WHERE key = 'total_ai_chats'", (new_count,))
                                    else:
                                        db_cursor.execute("INSERT INTO settings (key, value) VALUES ('total_ai_chats', '1')")
                                    db_conn.commit()
                                    db_conn.close()
                                except Exception:
                                    pass 
                                st.rerun()
                            except Exception as e:
                                st.error(f"Lỗi gọi API: {e}.")