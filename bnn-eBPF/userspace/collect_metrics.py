# Read BPF metrics & compute scores

import math

tp, fp, tn, fn, pkt, drop, total_ns = ... # read from bpftool

accuracy = (tp + tn) / pkt
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
latency_ns = total_ns / pkt

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Avg latency (ns):", latency_ns)
