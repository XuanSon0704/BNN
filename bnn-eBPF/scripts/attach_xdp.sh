#!/bin/bash
DEV=eth0
ip link set dev $DEV xdp obj ebpf/bnn_ebpf.o sec xdp
