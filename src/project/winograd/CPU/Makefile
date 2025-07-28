CC = gcc
CFLAGS = -std=c11 -O3 -fopenmp
TARGET = winograd
SOURCES = main.c naive_conv.c winograd_conv.c

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean