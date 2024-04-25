import sys  

def main():
    while (line := sys.stdin.readline()):
        W, H = line.strip().split(" ")
        _, halide_time, exo_time = sys.stdin.readline().strip().split(" ")
        sys.stdin.readline()

        print(f"{W}x{H}\t{halide_time}\t{exo_time}")

if __name__ == "__main__":
    main()
