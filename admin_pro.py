import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime
import time
import io
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 0. HÀM BỔ TRỢ & DATABASE MIGRATION NÂNG CẤP
# ==========================================
def upgrade_database():
    """Hàm tự động thêm các field/table mới mà không làm mất dữ liệu cũ"""
    conn = sqlite3.connect('autovision.db')
    c = conn.cursor()
    
    # 1. Nâng cấp bảng users (Thêm field trong try-except để không lỗi nếu field đã có)
    columns_to_add = [
        ("created_at", "TEXT", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("last_login_at", "TEXT", ""),
        ("is_active", "INTEGER", 1),
        ("total_valuations", "INTEGER", 0)
    ]
    for col_name, col_type, default_val in columns_to_add:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type} DEFAULT '{default_val}'")
        except sqlite3.OperationalError:
            pass # Cột đã tồn tại thì bỏ qua
            
    # 2. Tạo bảng Settings
    c.execute('''CREATE TABLE IF NOT EXISTS settings 
                 (key TEXT PRIMARY KEY, value TEXT)''')
                 
    # 3. [XÓA BẢNG VEHICLES]
    c.execute("DROP TABLE IF EXISTS vehicles")
    
    # 4. TẠO BẢNG LƯU TRỮ LỊCH SỬ CHAT RIÊNG CHO TỪNG USER
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                 (chat_id TEXT PRIMARY KEY, username TEXT, title TEXT, messages TEXT, pinned INTEGER, timestamp REAL)''')
    
    # Insert default settings nếu chưa có
    try:
        c.execute("INSERT INTO settings (key, value) VALUES ('ai_chat_enabled', '1')")
        c.execute("INSERT INTO settings (key, value) VALUES ('system_name', 'AutoVision Ultimate')")
        c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('total_ai_chats', '0')")
    except:
        pass
        

    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def get_db_connection():
    return sqlite3.connect('autovision.db', check_same_thread=False)

# ==========================================
# GIAO DIỆN CHÍNH CỦA ADMIN PRO
# ==========================================
def render_admin_dashboard():
    # Chạy kịch bản nâng cấp DB ngầm (chỉ chạy 1 lần khi admin vào)
    upgrade_database()
    
    conn = get_db_connection()
    
    st.title("⚙️ HỆ THỐNG QUẢN TRỊ AUTOVISION PRO")
    
    # Tạo các Tab theo đúng yêu cầu
    tab_dash, tab_users, tab_history, tab_api, tab_settings = st.tabs([
        "📊 Dashboard", "👥 User Pro", "📜 Lịch Sử ĐG", "🤖 Quản Lý AI/API", "⚙️ Settings"
    ])
    
    # ---------------------------------------------------------
    # III. DASHBOARD
    # ---------------------------------------------------------
    with tab_dash:
        st.subheader("1. Tổng Quan Chỉ Số (KPIs)")
        # Lấy data
        df_hist = pd.read_sql("SELECT * FROM history", conn)
        df_users = pd.read_sql("SELECT * FROM users", conn)
        
        col1, col2, col3, col4 = st.columns(4)
        total_users = len(df_users)
        total_vals = len(df_hist)
        total_revenue = df_hist['final_price'].sum() if not df_hist.empty else 0
        
        col1.metric("Tổng số User", total_users)
        col2.metric("Lượt Định Giá", total_vals)
        col3.metric("Tổng Giá Trị (VNĐ)", f"{total_revenue:,.0f}")
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key = 'total_ai_chats'")
        ai_chats = c.fetchone()
        ai_count = int(ai_chats[0]) if ai_chats else 0
        col4.metric("Lượt AI Chat", ai_count)
        
        st.subheader("2. Biểu Đồ Phân Tích")
        if not df_hist.empty:
            df_hist['date'] = pd.to_datetime(df_hist['timestamp']).dt.date
            
            c1, c2 = st.columns(2)
            with c1:
                # Line chart tăng trưởng
                daily_counts = df_hist.groupby('date').size().reset_index(name='counts')
                fig_line = px.line(daily_counts, x='date', y='counts', title='Tăng trưởng định giá theo ngày', markers=True)
                st.plotly_chart(fig_line, use_container_width=True)
                
            with c2:
                # Pie chart role
                role_counts = df_users['role'].value_counts().reset_index()
                role_counts.columns = ['role', 'count']
                fig_pie = px.pie(role_counts, values='count', names='role', title='Phân bố User Role')
                st.plotly_chart(fig_pie, use_container_width=True)
                
            # Bar chart Top 10 xe
            top_cars = df_hist['car_name'].value_counts().head(10).reset_index()
            top_cars.columns = ['car_name', 'count']
            fig_bar = px.bar(top_cars, x='car_name', y='count', title='Top 10 xe được định giá nhiều nhất')
            st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------------------------------------------
    # I. USER MANAGEMENT PRO
    # ---------------------------------------------------------
    with tab_users:
        st.subheader("Quản Lý Người Dùng Cấp Cao")
        
        # Thêm user mới
        with st.expander("➕ Tạo User Mới"):
            # THÊM clear_on_submit=True ĐỂ TỰ ĐỘNG XÓA TRẮNG SAU KHI BẤM TẠO
            with st.form("new_user_form", clear_on_submit=True):
                u_name = st.text_input("Username", key="admin_new_u_name")
                u_pass = st.text_input("Password", type="password", key="admin_new_u_pass")
                u_role = st.selectbox("Role", ["user", "admin"])
                if st.form_submit_button("Tạo Tài Khoản"):
                    try:
                        c = conn.cursor()
                        c.execute("INSERT INTO users (username, password, role, created_at, is_active, total_valuations) VALUES (?, ?, ?, ?, ?, ?)",
                                  (u_name, make_hashes(u_pass), u_role, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 1, 0))
                        conn.commit()
                        st.success("Đã tạo user thành công!")
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
        
        # Bảng quản lý
        df_u = pd.read_sql("SELECT username, role, is_active, created_at, last_login_at, total_valuations FROM users", conn)
        st.dataframe(df_u, use_container_width=True)
        
        # Các thao tác CRUD
        c_edit, c_del = st.columns(2)
        with c_edit:
            st.write("🛠️ **Sửa thông tin / Đổi mật khẩu**")
            sel_user = st.selectbox("Chọn User để thao tác:", df_u['username'].tolist())
            new_pass = st.text_input("Mật khẩu mới (Bỏ trống nếu không đổi)", type="password", key="reset_pass")
            new_role = st.selectbox("Đổi Role:", ["Giữ nguyên", "user", "admin"])
            new_status = st.selectbox("Trạng thái (Khóa/Mở):", ["Giữ nguyên", "Hoạt động", "Khóa"])
            
            if st.button("💾 Cập nhật User"):
                c = conn.cursor()
                has_error = False
                has_update = False
                
                # 1. KIỂM TRA MẬT KHẨU TRÙNG
                if len(new_pass) > 0:
                    c.execute("SELECT password FROM users WHERE username = ?", (sel_user,))
                    current_pass = c.fetchone()[0]
                    
                    if current_pass == make_hashes(new_pass):
                        st.error("⚠️ Mật khẩu này đang được sử dụng! Vui lòng nhập mật khẩu khác.")
                        has_error = True
                    else:
                        c.execute("UPDATE users SET password = ? WHERE username = ?", (make_hashes(new_pass), sel_user))
                        has_update = True
                        
                # 2. CHỈ CẬP NHẬT TIẾP NẾU KHÔNG BỊ LỖI TRÙNG MẬT KHẨU
                if not has_error:
                    if new_role != "Giữ nguyên":
                        c.execute("UPDATE users SET role = ? WHERE username = ?", (new_role, sel_user))
                        has_update = True
                    if new_status != "Giữ nguyên":
                        s_val = 1 if new_status == "Hoạt động" else 0
                        c.execute("UPDATE users SET is_active = ? WHERE username = ?", (s_val, sel_user))
                        has_update = True
                        
                    if has_update:
                        conn.commit()
                        st.success("Cập nhật thành công!")
                        # Dọn rác khung nhập mật khẩu để nó trống không
                        if 'reset_pass' in st.session_state:
                            del st.session_state['reset_pass']
                        time.sleep(0.5)
                        st.rerun()
                    elif len(new_pass) == 0:
                        st.warning("Bạn chưa thay đổi thông tin nào!")

        with c_del:
            st.write("🗑️ **Xóa User (Soft Delete)**")
            st.warning("Hệ thống chỉ chuyển is_active = 0, không xóa dữ liệu vật lý.")
            if st.button("❌ Khóa / Soft Delete tài khoản đang chọn", type="primary"):
                c = conn.cursor()
                c.execute("UPDATE users SET is_active = 0 WHERE username = ?", (sel_user,))
                conn.commit()
                st.success(f"Đã khóa tài khoản {sel_user}.")
                time.sleep(0.5)
                st.rerun()

    # ---------------------------------------------------------
    # II. QUẢN LÝ LỊCH SỬ ĐỊNH GIÁ (Lọc & Export)
    # ---------------------------------------------------------
    with tab_history:
        st.subheader("Bộ Lọc Tìm Kiếm Nâng Cao")
        df_h = pd.read_sql("SELECT * FROM history ORDER BY id DESC", conn)
        
        if not df_h.empty:
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                filter_user = st.multiselect("Lọc theo User:", df_h['username'].unique())
            with col_f2:
                filter_car = st.text_input("Tìm Tên xe (Từ khóa):")
            with col_f3:
                min_p = float(df_h['final_price'].min())
                max_p = float(df_h['final_price'].max())
                filter_price = st.slider("Khoảng giá:", min_value=min_p, max_value=max_p, value=(min_p, max_p))
                
            # Áp dụng bộ lọc
            mask = pd.Series([True]*len(df_h))
            if filter_user: mask = mask & df_h['username'].isin(filter_user)
            if filter_car: mask = mask & df_h['car_name'].str.contains(filter_car, case=False, na=False)
            mask = mask & (df_h['final_price'] >= filter_price[0]) & (df_h['final_price'] <= filter_price[1])
            
            filtered_df = df_h[mask]
            st.dataframe(filtered_df, use_container_width=True)
            
            # Tính năng Xuất file (Export)
            st.write("📥 **Xuất Dữ Liệu:**")
            col_ex1, col_ex2, _ = st.columns([1,1,3])
            
            # Xuất CSV
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            col_ex1.download_button("Xuất Excel (CSV)", data=csv, file_name='lich_su_dinh_gia.csv', mime='text/csv')
            
            # Tính năng Xóa Record
            st.write("🗑️ **Xóa Record Lịch Sử:**")
            del_id = st.number_input("Nhập ID record cần xóa:", min_value=0, step=1)
            if st.button("Xóa Record ID"):
                c = conn.cursor()
                c.execute("DELETE FROM history WHERE id = ?", (del_id,))
                conn.commit()
                st.success("Đã xóa thành công!")
                time.sleep(0.5)
                st.rerun()

    # ---------------------------------------------------------
    # IV. QUẢN LÝ API & AI
    # ---------------------------------------------------------
    with tab_api:
        st.subheader("Cấu Hình Model AI Chat")
        
        # Đọc settings
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key = 'ai_chat_enabled'")
        ai_enabled = c.fetchone()
        is_ai_on = True if (ai_enabled and ai_enabled[0] == '1') else False
        
        toggle_ai = st.toggle("Bật / Tắt tính năng Trợ lý AI Chat trên toàn hệ thống", value=is_ai_on)
        if toggle_ai != is_ai_on:
            new_val = '1' if toggle_ai else '0'
            c.execute("UPDATE settings SET value = ? WHERE key = 'ai_chat_enabled'", (new_val,))
            conn.commit()
            st.success("Đã thay đổi trạng thái AI!")

        st.info("""
        ℹ️ **Thông tin API (Tham khảo)**:
        - **Model đang dùng**: `gemini-3-flash-preview` (Text) & `YOLOv8` (Vision)
        - **Chi phí**: Đang sử dụng Tier Miễn phí API của Google (Rate limit: ~15 RPM cho bản Preview).
        - **Tốc độ**: Phiên bản Flash-Preview tối ưu hóa tốc độ phản hồi siêu nhanh so với các bản Pro cũ.
        - **Key**: Key được bảo mật an toàn tuyệt đối qua hệ thống cấu hình nội bộ, không lưu trong Database.
        """)

    # ---------------------------------------------------------
    # V. SYSTEM SETTINGS
    # ---------------------------------------------------------
    with tab_settings:
        st.subheader("Cấu Hình Hệ Thống Chung")
        c = conn.cursor()
        
        # 1. Đọc tên hệ thống
        c.execute("SELECT value FROM settings WHERE key = 'system_name'")
        sys_name = c.fetchone()
        curr_name = sys_name[0] if sys_name else "AutoVision Ultimate"
        
        # 2. Đọc giới hạn định giá từ Database (để không bị reset về 0)
        c.execute("SELECT value FROM settings WHERE key = 'valuation_limit'")
        limit_db = c.fetchone()
        curr_limit = int(limit_db[0]) if limit_db else 0
        
        with st.form("settings_form"):
            new_sys_name = st.text_input("Tên hệ thống hiển thị:", value=curr_name)
            limit_val = st.number_input("Giới hạn lượt định giá/ngày (0 = Vô hạn):", value=curr_limit, min_value=0, step=1)
            
            if st.form_submit_button("Lưu Cài Đặt"):
                # Lưu cả tên và giới hạn vào Database
                c.execute("UPDATE settings SET value = ? WHERE key = 'system_name'", (new_sys_name,))
                # Dùng INSERT OR REPLACE/UPDATE để đảm bảo ghi được số giới hạn
                c.execute("SELECT key FROM settings WHERE key='valuation_limit'")
                if c.fetchone():
                    c.execute("UPDATE settings SET value = ? WHERE key = 'valuation_limit'", (str(limit_val),))
                else:
                    c.execute("INSERT INTO settings (key, value) VALUES ('valuation_limit', ?)", (str(limit_val),))
                
                conn.commit()
                st.success("Lưu cài đặt thành công!")
                time.sleep(0.5)
                st.rerun()
# Dòng này để test riêng file nếu chạy trực tiếp
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_admin_dashboard()