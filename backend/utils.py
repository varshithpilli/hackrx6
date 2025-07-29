def log_time(label, start, end):
    with open("timings.log", "a") as f:
        f.write(f"{label}: {end - start:.2f} seconds\n")
        