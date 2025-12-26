import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Đường dẫn map 
MAP_PATH = "/sys/fs/bpf/metrics_map"

def read_metrics():
    # Dùng bpftool dump map
    cmd = ["sudo", "bpftool", "map", "dump", "pinned", MAP_PATH]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0: return None
    
    try:
        output = res.stdout
        tp = int(re.search(r'"tp":\s*(\d+)', output).group(1))
        fp = int(re.search(r'"fp":\s*(\d+)', output).group(1))
        tn = int(re.search(r'"tn":\s*(\d+)', output).group(1))
        fn = int(re.search(r'"fn":\s*(\d+)', output).group(1))
        pkt_count = int(re.search(r'"pkt_count":\s*(\d+)', output).group(1))
        total_ns = int(re.search(r'"total_proc_ns":\s*(\d+)', output).group(1))
        return tp, fp, tn, fn, pkt_count, total_ns
    except:
        return 0,0,0,0,0,0

def calc_metrics(tp, fp, tn, fn):
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp+fp+tn+fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return acc, prec, rec, f1

def main():
    history = {'time':[], 'acc':[], 'f1':[], 'latency':[], 'pps':[]}
    start_time = time.time()
    last_pkt = 0
    last_time = start_time

    try:
        while True:
            tp, fp, tn, fn, pkts, total_ns = read_metrics()
            curr_time = time.time()
            
            acc, prec, rec, f1 = calc_metrics(tp, fp, tn, fn)
            
            # Latency (Average ns per packet)
            avg_lat = total_ns / pkts if pkts > 0 else 0
            
            # Throughput (PPS) - Instantaneous
            dt = curr_time - last_time
            d_pkts = pkts - last_pkt
            pps = d_pkts / dt if dt > 0 else 0
            
            # Update history
            t_offset = curr_time - start_time
            history['time'].append(t_offset)
            history['acc'].append(acc)
            history['f1'].append(f1)
            history['latency'].append(avg_lat)
            history['pps'].append(pps)
            
            print(f"[{t_offset:.1f}s] Acc: {acc:.2f} | F1: {f1:.2f} | Latency: {avg_lat:.0f}ns | PPS: {pps:.0f}")
            
            last_pkt = pkts
            last_time = curr_time
            time.sleep(1) # Refresh every 1s

    except KeyboardInterrupt:
        print("\nStopping and plotting...")
        
        # Plotting
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        
        # Accuracy
        ax[0,0].plot(history['time'], history['acc'], 'b-')
        ax[0,0].set_title("Kernel Accuracy")
        ax[0,0].set_ylim(0, 1.1)
        
        # F1
        ax[0,1].plot(history['time'], history['f1'], 'r-')
        ax[0,1].set_title("Kernel F1-Score")
        ax[0,1].set_ylim(0, 1.1)
        
        # Latency
        ax[1,0].plot(history['time'], history['latency'], 'g-')
        ax[1,0].set_title("Avg Latency (ns)")
        
        # PPS
        ax[1,1].plot(history['time'], history['pps'], 'k-')
        ax[1,1].set_title("Throughput (PPS)")
        
        plt.tight_layout()
        plt.savefig("../results/kernel_metrics.png")
        print("Saved plot to ../results/kernel_metrics.png")

if __name__ == "__main__":
    main()