import sys
import os
import shutil
import platform
import subprocess
import tkinter as tk
import webbrowser

def show_error_dialog(title, message, error_code, help_link, is_soft_fail=False):
    dialog = tk.Tk()
    dialog.title(title)
    dialog.geometry("550x250")
    dialog.resizable(False, False)
    
    # center
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f'+{x}+{y}')
    
    user_proceeds = [False]
    
    frame = tk.Frame(dialog, padx=20, pady=20)
    frame.pack(expand=True, fill=tk.BOTH)
    
    msg_lbl = tk.Label(frame, text=message, justify=tk.LEFT, wraplength=500, font=("Helvetica", 11))
    msg_lbl.pack(anchor=tk.W, pady=(0, 10))
    
    code_lbl = tk.Label(frame, text=f"Error Code: {error_code}", font=("Helvetica", 10, "bold"))
    code_lbl.pack(anchor=tk.W, pady=(0, 15))
    
    btn_frame = tk.Frame(frame)
    btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
    
    def copy_log():
        log_text = f"Error Code: {error_code}\nMessage: {message}\n"
        log_text += f"OS: {platform.system()} {platform.release()}\n"
        log_text += f"Python: {sys.version}\n"
        dialog.clipboard_clear()
        dialog.clipboard_append(log_text)
        dialog.update()
        
    def open_help():
        webbrowser.open(help_link)
        
    def quit_app():
        dialog.destroy()
        if not is_soft_fail:
            sys.exit(1)
            
    def proceed():
        user_proceeds[0] = True
        dialog.destroy()
        
    tk.Button(btn_frame, text="Copy Log", command=copy_log, width=12).pack(side=tk.LEFT, padx=5)
    if help_link:
        tk.Button(btn_frame, text="Information Guide", command=open_help, width=15).pack(side=tk.LEFT, padx=5)
        
    if is_soft_fail:
        tk.Button(btn_frame, text="Proceed Anyway", command=proceed, width=15).pack(side=tk.RIGHT, padx=5)
        tk.Button(btn_frame, text="Exit Application", command=quit_app, width=15).pack(side=tk.RIGHT, padx=5)
    else:
        tk.Button(btn_frame, text="Exit Application", command=quit_app, width=15).pack(side=tk.RIGHT, padx=5)
        
    dialog.protocol("WM_DELETE_WINDOW", quit_app)
    dialog.mainloop()
    
    if not is_soft_fail and not user_proceeds[0]:
        sys.exit(1)

def check_nvidia_gpu_present():
    try:
        sys_plat = platform.system()
        if sys_plat == 'Windows':
            try:
                output = subprocess.check_output('wmic path win32_VideoController get name', shell=True, text=True, stderr=subprocess.DEVNULL).lower()
            except subprocess.CalledProcessError:
                output = subprocess.check_output('powershell -Command "Get-CimInstance -ClassName Win32_VideoController | Select-Object -Property Name"', shell=True, text=True, stderr=subprocess.DEVNULL).lower()
            return 'nvidia' in output
        elif sys_plat == 'Linux':
            try:
                output = subprocess.check_output('lspci', shell=True, text=True, stderr=subprocess.DEVNULL).lower()
                return 'nvidia' in output or 'geforce' in output
            except Exception:
                pass
    except Exception:
        pass
    return False

def check_apple_silicon_present():
    try:
        if platform.system() == 'Darwin':
            output = subprocess.check_output('system_profiler SPHardwareDataType', shell=True, text=True, stderr=subprocess.DEVNULL).lower()
            return 'apple' in output and ('m1' in output or 'm2' in output or 'm3' in output or 'silicon' in output)
    except Exception:
        pass
    return False

def run_preflight_checks():
    # 1. Python version check
    if sys.version_info < (3, 9):
        show_error_dialog(
            title="Python Version Error",
            message=f"Unsupported Python version {sys.version.split()[0]} detected. Ultimate Vocal Remover requires Python 3.9 or higher. Please upgrade your Python installation to continue.",
            error_code="ERR_PYTHON_VERSION",
            help_link="https://www.python.org/downloads/",
            is_soft_fail=False
        )
        
    # Determine base path for local bins
    if getattr(sys, 'frozen', False):
        BASE_PATH = sys._MEIPASS
    else:
        BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    # 2. FFmpeg check
    ffmpeg_name = 'ffmpeg.exe' if platform.system() == 'Windows' else 'ffmpeg'
    ffmpeg_found = shutil.which('ffmpeg') or os.path.isfile(os.path.join(BASE_PATH, ffmpeg_name))
    if not ffmpeg_found:
        show_error_dialog(
            title="Missing Binary",
            message="FFmpeg could not be found. UVR relies on FFmpeg for processing non-wav audio files.\nPlease install FFmpeg or place the executable in the correct directory.",
            error_code="ERR_MISSING_FFMPEG",
            help_link="https://www.wikihow.com/Install-FFmpeg-on-Windows" if platform.system() == 'Windows' else "https://ffmpeg.org/download.html",
            is_soft_fail=False
        )

    # 3. Rubber Band check
    rubberband_name = 'rubberband.exe' if platform.system() == 'Windows' else 'rubberband'
    rb_path_root = os.path.join(BASE_PATH, rubberband_name)
    rb_path_lib = os.path.join(BASE_PATH, 'lib_v5', rubberband_name)
    rb_found = shutil.which('rubberband') or os.path.isfile(rb_path_root) or os.path.isfile(rb_path_lib)
    if not rb_found:
        show_error_dialog(
            title="Missing Binary",
            message="Rubber Band library could not be found. UVR uses the Rubber Band library for the sound stretch and pitch shift tool.\nPlease install it or place the executable in the application directory.",
            error_code="ERR_MISSING_RUBBERBAND",
            help_link="https://breakfastquay.com/rubberband/",
            is_soft_fail=False
        )
        
    # 4. GPU (CUDA/MPS) check
    try:
        import torch
    except ImportError:
        show_error_dialog(
            title="Missing Dependency",
            message="PyTorch could not be loaded. Please ensure the application dependencies are installed correctly.",
            error_code="ERR_MISSING_PYTORCH",
            help_link="https://github.com/Anjok07/ultimatevocalremovergui#troubleshooting",
            is_soft_fail=False
        )
        return
        
    nvidia_present = check_nvidia_gpu_present()
    apple_silicon = check_apple_silicon_present()
    
    if nvidia_present:
        cuda_avail = torch.cuda.is_available()
        if not cuda_avail:
            show_error_dialog(
                title="Incompatible Hardware Acceleration",
                message="An NVIDIA GPU is present, but CUDA is unavailable or installed drivers are incompatible. You can proceed in CPU-only mode, or update your drivers.",
                error_code="WARN_CUDA_UNAVAILABLE",
                help_link="https://github.com/Anjok07/ultimatevocalremovergui#troubleshooting",
                is_soft_fail=True
            )
    elif apple_silicon:
        mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if not mps_avail:
            show_error_dialog(
                title="Incompatible Hardware Acceleration",
                message="Apple Silicon detected, but MPS is unavailable. Your PyTorch version might be incompatible. You can proceed in CPU-only mode, or update PyTorch.",
                error_code="WARN_MPS_UNAVAILABLE",
                help_link="https://github.com/Anjok07/ultimatevocalremovergui#troubleshooting",
                is_soft_fail=True
            )

if __name__ == '__main__':
    run_preflight_checks()
