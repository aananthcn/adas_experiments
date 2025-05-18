import subprocess
import csv
import re
import os

def parse_output(output):
    """Parse the output of vector_mac to extract timing metrics."""
    metrics = {
        "CPU Initialization Time": None,
        "CPU MAC Time": None,
        "GPU Data Transfer Time (Write)": None,
        "GPU MAC Computation Time": None,
        "GPU Data Transfer Time (Read)": None,
        "CPU Verification Time": None,
        "Verification Status": "Failed"
    }
    
    # Split output into lines
    lines = output.splitlines()
    
    # Regular expressions for parsing
    time_pattern = re.compile(r"(.+?)(?:\s*\(\d+\s*operations\))?:\s*([\d.]+)\s*ms")
    verification_pattern = re.compile(r"Verification (passed|failed)")
    
    for line in lines:
        # Parse timing metrics
        time_match = time_pattern.match(line)
        if time_match:
            metric, value = time_match.groups()
            metric = metric.strip()
            # Map GPU MAC computation times to a single key
            if metric.startswith("GPU Sequential MAC Computation Time") or metric.startswith("GPU Parallel MAC Computation Time"):
                metrics["GPU MAC Computation Time"] = float(value)
            elif metric in metrics:
                metrics[metric] = float(value)
        
        # Parse verification status
        verification_match = verification_pattern.match(line)
        if verification_match:
            metrics["Verification Status"] = verification_match.group(1).capitalize()
    
    return metrics

def run_vector_mac(megapixels, num_operations, mode):
    """Run the vector_mac binary and return parsed metrics."""
    cmd = ["./vector_mac", "-p", str(megapixels), "-o", str(num_operations), "-m", mode]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout + result.stderr
        metrics = parse_output(output)
        metrics["Megapixels"] = megapixels
        metrics["Num Operations"] = num_operations
        metrics["Mode"] = "Parallel" if mode == "P" else "Sequential"
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"Error running command {cmd}: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error running command {cmd}: {str(e)}")
        return None

def main():
    # Define parameter ranges
    megapixels_range = range(2, 11, 2)  # 2, 4, 6, 8, 10
    num_operations_range = range(5, 21, 5)  # 5, 10, 15, 20
    modes = ["S", "P"]
    
    # CSV output file
    output_file = "vector_mac_results.csv"
    
    # CSV headers
    headers = [
        "Megapixels",
        "Num Operations",
        "Mode",
        "CPU Initialization Time (ms)",
        "CPU MAC Time (ms)",
        "GPU Data Transfer Time (Write) (ms)",
        "GPU MAC Computation Time (ms)",
        "GPU Data Transfer Time (Read) (ms)",
        "CPU Verification Time (ms)",
        "Verification Status"
    ]
    
    # Collect results
    results = []
    for megapixels in megapixels_range:
        for num_operations in num_operations_range:
            for mode in modes:
                print(f"Running: megapixels={megapixels}, num_operations={num_operations}, mode={mode}")
                metrics = run_vector_mac(megapixels, num_operations, mode)
                if metrics:
                    results.append(metrics)
    
    # Write results to CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "Megapixels": result["Megapixels"],
                "Num Operations": result["Num Operations"],
                "Mode": result["Mode"],
                "CPU Initialization Time (ms)": result["CPU Initialization Time"],
                "CPU MAC Time (ms)": result["CPU MAC Time"],
                "GPU Data Transfer Time (Write) (ms)": result["GPU Data Transfer Time (Write)"],
                "GPU MAC Computation Time (ms)": result["GPU MAC Computation Time"],
                "GPU Data Transfer Time (Read) (ms)": result["GPU Data Transfer Time (Read)"],
                "CPU Verification Time (ms)": result["CPU Verification Time"],
                "Verification Status": result["Verification Status"]
            })
    
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    # Check if vector_mac binary exists
    if not os.path.isfile("./vector_mac"):
        print("Error: vector_mac binary not found in current directory.")
    else:
        main()