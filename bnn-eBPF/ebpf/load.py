import subprocess
import os
import sys
import numpy as np

BPF_FS = "/sys/fs/bpf"
WEIGHTS_PATH = "../results/bnn_weights.npz"

def get_map_id(name):
    """Tìm ID của map từ tên"""
    cmd = ["sudo", "bpftool", "map", "show"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.splitlines():
        parts = line.split()
        # Tìm dòng chứa "name <map_name>"
        if len(parts) >= 4 and parts[2] == "name" and parts[3] == name:
            return parts[0].strip(":")
    return None

def pin_map(map_name):
    """Pin map từ RAM ra file system nếu chưa pin"""
    pinned_path = os.path.join(BPF_FS, map_name)
    
    # Nếu file đã tồn tại, xóa đi để pin lại cái mới nhất (tránh lỗi cũ đè mới)
    if os.path.exists(pinned_path):
        subprocess.run(["sudo", "rm", pinned_path], check=True)
        print(f"  [Info] Removed old pin: {pinned_path}")

    map_id = get_map_id(map_name)
    if not map_id:
        print(f"  [Error] Map '{map_name}' not found in Kernel. Is XDP loaded?")
        return False

    cmd = ["sudo", "bpftool", "map", "pin", "id", map_id, pinned_path]
    subprocess.run(cmd, check=True)
    print(f"  [OK] Pinned '{map_name}' (ID: {map_id}) -> {pinned_path}")
    return True

# --- Phần hàm update weights ---
def u64_to_hex_list(val, is_signed=False):
    val_int = int(val)
    byte_data = val_int.to_bytes(8, byteorder='little', signed=is_signed)
    return [f"0x{b:02x}" for b in byte_data]

def update_map_data(map_name, data, is_packed=False, is_signed=False):
    # --- SỬA LỖI: Thêm dòng định nghĩa này ---
    pinned_path = os.path.join(BPF_FS, map_name)
    # ----------------------------------------
    
    print(f"  Updating data for {map_name}...")
    rows = data.shape[0]
    for i in range(rows):
        key_hex = [f"0x{b:02x}" for b in i.to_bytes(4, 'little')]
        val_hex = []
        if is_packed:
            row_vals = data[i]
            if isinstance(row_vals, np.ndarray):
                for v in row_vals: val_hex.extend(u64_to_hex_list(v, is_signed))
            else: val_hex.extend(u64_to_hex_list(row_vals, is_signed))
        else:
            val_hex.extend(u64_to_hex_list(data[i], is_signed))

        cmd = ["sudo", "bpftool", "map", "update", "pinned", pinned_path, "key"] + key_hex + ["value"] + val_hex
        subprocess.run(cmd, capture_output=True)

def main():
    print(">>> STEP 1: Pinning Maps...")
    # Pin metrics_map (quan trọng cho Monitor)
    pin_map("metrics_map")
    
    # Pin các map Weights
    maps_to_pin = ["threshold_map", "w1_map", "t1_map", "w2_map", "t2_map", "w_out_map"]
    for m in maps_to_pin:
        pin_map(m)

    print("\n>>> STEP 2: Loading Weights...")
    if not os.path.exists(WEIGHTS_PATH):
        print("Weight file not found!")
        return
    data = np.load(WEIGHTS_PATH)

    if 'input_means' in data: update_map_data("threshold_map", data['input_means'])
    elif 'thresholds' in data: update_map_data("threshold_map", data['thresholds'])

    update_map_data("w1_map", data['w1'], is_packed=True)
    update_map_data("t1_map", data['t1'], is_signed=True)
    update_map_data("w2_map", data['w2'], is_packed=True)
    update_map_data("t2_map", data['t2'], is_signed=True)
    update_map_data("w_out_map", data['wout'], is_packed=True)
    
    print("\n>>> SETUP COMPLETE. Ready for Monitor & Traffic Gen.")

if __name__ == "__main__":
    main()