import platform

# grid_ocr imports WinRT at module load, so its tests can only run on Windows.
collect_ignore_glob = ["test_*.py"] if platform.system() != "Windows" else []
