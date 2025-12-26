import pandas as pd
from scapy.all import sendp, IP, TCP, Ether, Raw
import time
import sys
import math

# CẤU HÌNH
CSV_PATH = "/home/sonnguyen/workspace/BNN/bnn-eBPF/userspace/DDOS2017.csv"
IFACE = "lo"
PPS_TARGET = 1000 

def generate_traffic():
    # Đọc dữ liệu
    try:
        df = pd.read_csv(CSV_PATH, nrows=5000)
    except FileNotFoundError:
        print(f"Error: File {CSV_PATH} not found.")
        return

    df.columns = df.columns.str.strip()
    print(f"Start sending traffic on {IFACE}...")
    
    count = 0
    start_time = time.time()
    
    for index, row in df.iterrows():
        try:
            # --- FIX 1: Dùng 'Average Packet Size' thay vì 'Total Length' ---
            # Total Length là tổng cả dòng chảy, dùng cho 1 gói tin sẽ bị quá khổ (Errno 90)
            pkt_len_val = row.get('Average Packet Size', 64)
            
            # Xử lý giá trị NaN hoặc vô cực
            if pd.isna(pkt_len_val) or math.isinf(pkt_len_val):
                pkt_len = 64
            else:
                pkt_len = int(pkt_len_val)

            # Kẹp giá trị (Safety Clamp) để tránh Errno 90
            # IP Header (20) + TCP Header (20) = 40 bytes tối thiểu
            # MTU an toàn thường là 1400-1500 bytes
            if pkt_len < 40: pkt_len = 40 
            if pkt_len > 1400: pkt_len = 1400 
            
            # --- FIX 2: Xử lý TCP Window Size (struct.error) ---
            win_val = row.get('Init_Win_bytes_forward', 8192)
            if pd.isna(win_val) or math.isinf(win_val):
                win_size = 8192
            else:
                win_size = int(win_val)
            
            # Kẹp giá trị trong khoảng 2 bytes (0 - 65535)
            win_size = max(0, min(65535, win_size))
            
            # --- LABELING ---
            label_str = row.get('Label', 'BENIGN')
            is_ddos = 1 if label_str == 'DDoS' else 0
            payload_marker = bytes([is_ddos]) 
            
            # --- TẠO GÓI TIN ---
            # Payload len = Tổng len - (Ethernet + IP + TCP headers)
            # Ether=14, IP=20, TCP=20 -> Total headers ~ 54
            payload_len = max(0, pkt_len - 54)
            payload = payload_marker + b'\x00' * (payload_len - 1)
            
            # Lưu ý: Scapy tự tính Total Length trong IP header dựa trên payload
            # Ta chỉ cần tạo payload đủ dài là được.
            packet = Ether() / IP(dst="127.0.0.1") / TCP(dport=80, window=win_size) / Raw(load=payload)
            
            # --- GỬI ---
            sendp(packet, iface=IFACE, verbose=False)
            count += 1
            
            # Điều tốc
            if count % 100 == 0:
                elapsed = time.time() - start_time
                if elapsed == 0: elapsed = 0.001
                current_pps = count / elapsed
                
                if current_pps > PPS_TARGET:
                    sleep_time = (count / PPS_TARGET) - elapsed
                    if sleep_time > 0: time.sleep(sleep_time)
                    
        except Exception as e:
            # In lỗi gọn hơn để dễ debug
            print(f"Skipping row {index}: {e}")
            continue

    print(f"Done. Sent {count} packets.")

if __name__ == "__main__":
    generate_traffic()