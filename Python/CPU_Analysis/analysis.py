import psutil
import platform

def print_system_resource_usage():
    """Prints a well-formatted output of current system resource usage."""

    # System information
    print("-" * 40)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Node Name: {platform.node()}")
    print(f"Processor: {platform.processor()}")
    print("-" * 40)

    # CPU usage
    print("CPU Usage:")
    print(f"  - Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    for i, percent in enumerate(cpu_percent):
        print(f"  - Core {i}: {percent}%")
    print(f"  - Total: {psutil.cpu_percent()}%")

    # Memory usage
    print("-" * 40)
    print("Memory Usage:")
    virtual_memory = psutil.virtual_memory()
    print(f"  - Total: {virtual_memory.total / (1024 ** 3):.1f} GB")
    print(f"  - Available: {virtual_memory.available / (1024 ** 3):.1f} GB")
    print(f"  - Used: {virtual_memory.used / (1024 ** 3):.1f} GB ({virtual_memory.percent}%)")

    # Disk usage
    print("-" * 40)
    print("Disk Usage:")
    for partition in psutil.disk_partitions():
        try:
            disk = psutil.disk_usage(partition.mountpoint)
            print(f"  - Drive: {partition.device}")
            print(f"    - Mount Point: {partition.mountpoint}")
            print(f"    - File System: {partition.fstype}")
            print(f"    - Total Size: {disk.total / (1024 ** 3):.1f} GB")
            print(f"    - Used: {disk.used / (1024 ** 3):.1f} GB ({disk.percent}%)")
            print(f"    - Free: {disk.free / (1024 ** 3):.1f} GB")
        except PermissionError:
            print(f"  - Drive: {partition.device} (Permissions Error)")
    print("-" * 40)

if __name__ == "__main__":
    print_system_resource_usage()