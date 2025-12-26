CLANG ?= clang
CFLAGS = -O2 -g -Wall -target bpf

all:
	$(CLANG) $(CFLAGS) -c BNN.c -o BNN.o

clean:
	rm -f BNN.o
