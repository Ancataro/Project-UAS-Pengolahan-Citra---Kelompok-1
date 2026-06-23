import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Import pipeline functions
from main import main as run_detection
from add_missing_data import interpolate_missing_data
from visuallize import visualize_results

class ThreadsafeLogWriter(object):
    def __init__(self, msg_queue, is_stderr=False):
        self.msg_queue = msg_queue
        self.is_stderr = is_stderr

    def write(self, str_val):
        if str_val:
            self.msg_queue.put(("log", str_val))

    def flush(self):
        pass

class ALPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionPlate")
        self.root.geometry("800x650")
        self.root.configure(bg="#121212")
        
        # Queue for thread-safe UI updates
        self.msg_queue = queue.Queue()
        
        # Configure ttk styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TProgressbar", thickness=15, troughcolor="#2c2c2c", background="#00adb5")
        
        # Default Variables
        self.video_path = tk.StringVar(value=os.path.abspath("./sample.mp4"))
        self.model_path = tk.StringVar(value=os.path.abspath("./models/license_plate_detector.pt"))
        self.output_video_path = tk.StringVar(value=os.path.abspath("./output.mp4"))
        
        # Setup UI
        self.setup_ui()
        
        # Redirect stdout and stderr to the queue
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = ThreadsafeLogWriter(self.msg_queue, is_stderr=False)
        sys.stderr = ThreadsafeLogWriter(self.msg_queue, is_stderr=True)
        
        # Start queue poller
        self.root.after(100, self.process_queue)
        
    def setup_ui(self):
        # Header Title
        title_lbl = tk.Label(self.root, text="VisionPlate", 
                             font=("Helvetica", 14, "bold"), fg="#eeeeee", bg="#121212", pady=15)
        title_lbl.pack()
        
        # Input Configuration Card
        input_frame = tk.LabelFrame(self.root, text=" Konfigurasi Input & Output ", font=("Helvetica", 10, "bold"),
                                    fg="#00adb5", bg="#1e1e1e", bd=1, relief="solid", padx=15, pady=15)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Video Selector Row
        tk.Label(input_frame, text="Pilih Video Kendaraan:", fg="#eeeeee", bg="#1e1e1e", font=("Helvetica", 9)).grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(input_frame, textvariable=self.video_path, width=65, bg="#2c2c2c", fg="#eeeeee", insertbackground="white", bd=1).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(input_frame, text="Telusuri...", command=self.browse_video, bg="#3a3a3a", fg="#eeeeee", activebackground="#555555", activeforeground="white", bd=0, padx=10).grid(row=0, column=2, pady=5)
        
        # Model Selector Row
        tk.Label(input_frame, text="Pilih Model YOLOv8:", fg="#eeeeee", bg="#1e1e1e", font=("Helvetica", 9)).grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(input_frame, textvariable=self.model_path, width=65, bg="#2c2c2c", fg="#eeeeee", insertbackground="white", bd=1).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(input_frame, text="Telusuri...", command=self.browse_model, bg="#3a3a3a", fg="#eeeeee", activebackground="#555555", activeforeground="white", bd=0, padx=10).grid(row=1, column=2, pady=5)

        # Output Video Row
        tk.Label(input_frame, text="Simpan Video Hasil ke:", fg="#eeeeee", bg="#1e1e1e", font=("Helvetica", 9)).grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(input_frame, textvariable=self.output_video_path, width=65, bg="#2c2c2c", fg="#eeeeee", insertbackground="white", bd=1).grid(row=2, column=1, padx=10, pady=5)
        tk.Button(input_frame, text="Telusuri...", command=self.browse_output, bg="#3a3a3a", fg="#eeeeee", activebackground="#555555", activeforeground="white", bd=0, padx=10).grid(row=2, column=2, pady=5)
        
        # Actions Panel
        control_frame = tk.Frame(self.root, bg="#121212")
        control_frame.pack(fill="x", padx=20, pady=5)
        
        self.start_btn = tk.Button(control_frame, text="MULAI PROSES DETEKSI", command=self.start_processing, 
                                   font=("Helvetica", 10, "bold"), bg="#00adb5", fg="#ffffff", 
                                   activebackground="#008c9e", activeforeground="white", bd=0, pady=8, width=25)
        self.start_btn.pack(side="left", padx=5)
        
        self.play_btn = tk.Button(control_frame, text="PUTAR VIDEO HASIL", command=self.play_output_video, 
                                  font=("Helvetica", 10, "bold"), bg="#3f72af", fg="#ffffff", 
                                  activebackground="#366296", activeforeground="white", bd=0, pady=8, width=20, state="disabled")
        self.play_btn.pack(side="left", padx=5)


        # Progress Status
        progress_frame = tk.Frame(self.root, bg="#121212")
        progress_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_lbl = tk.Label(progress_frame, text="Status: Siap untuk memproses", fg="#eeeeee", bg="#121212", font=("Helvetica", 9, "italic"))
        self.status_lbl.pack(anchor="w", pady=2)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', style="TProgressbar")
        self.progress_bar.pack(fill="x", pady=5)

        # Logs Window Card
        log_frame = tk.LabelFrame(self.root, text=" Log Konsol Pemrosesan ", font=("Helvetica", 10, "bold"),
                                  fg="#00adb5", bg="#121212", bd=1, relief="solid")
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.log_text = tk.Text(log_frame, bg="#1e1e1e", fg="#eeeeee", state="disabled", wrap="word", 
                                font=("Consolas", 9), insertbackground="white")
        self.log_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
    def browse_video(self):
        file = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")])
        if file:
            self.video_path.set(os.path.abspath(file))
            
    def browse_model(self):
        file = filedialog.askopenfilename(filetypes=[("YOLO Model weights", "*.pt")])
        if file:
            self.model_path.set(os.path.abspath(file))
            
    def browse_output(self):
        file = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("Video Files", "*.mp4")])
        if file:
            self.output_video_path.set(os.path.abspath(file))
            
    def queue_progress(self, current, total, stage):
        self.msg_queue.put(("progress", current, total, stage))
        
    def process_queue(self):
        while not self.msg_queue.empty():
            try:
                msg_type, *msg_data = self.msg_queue.get_nowait()
                if msg_type == "log":
                    str_val = msg_data[0]
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", str_val)
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
                elif msg_type == "progress":
                    current, total, stage = msg_data
                    pct = int((current / total) * 100)
                    self.progress_bar['value'] = pct
                    self.status_lbl.configure(text=f"Status: {stage} ({current}/{total} frame - {pct}%)")
                elif msg_type == "status":
                    self.status_lbl.configure(text=f"Status: {msg_data[0]}")
                elif msg_type == "ui_state":
                    state = msg_data[0]
                    if state == "done":
                        self.play_btn.configure(state="normal")
                        self.start_btn.configure(state="normal")
                        messagebox.showinfo("Sukses", "Deteksi dan visualisasi plat nomor selesai dengan sukses!")
                    elif state == "error":
                        err_msg = msg_data[1]
                        messagebox.showerror("Error", f"Terjadi kesalahan saat memproses:\n{err_msg}")
                        self.start_btn.configure(state="normal")
            except queue.Empty:
                break
            except Exception as e:
                self.old_stdout.write(f"Queue error: {e}\n")
                break
                
        # Reschedule queue check
        self.root.after(100, self.process_queue)
        
    def start_processing(self):
        # Validasi berkas input
        v_path = self.video_path.get()
        m_path = self.model_path.get()
        out_path = self.output_video_path.get()
        
        if not os.path.exists(v_path):
            messagebox.showerror("Error", f"File video '{v_path}' tidak ditemukan!")
            return
        if not os.path.exists(m_path) and not m_path.endswith('license_plate_detector.pt'):
            messagebox.showerror("Error", f"File model '{m_path}' tidak ditemukan!")
            return
            
        self.start_btn.configure(state="disabled")
        self.play_btn.configure(state="disabled")
        self.progress_bar['value'] = 0
        
        # Jalankan di Thread latar belakang agar GUI tetap interaktif
        thread = threading.Thread(target=self.process_pipeline, args=(v_path, m_path, out_path))
        thread.daemon = True
        thread.start()
        
    def process_pipeline(self, v_path, m_path, out_path):
        try:
            # Tahap 1: Deteksi dan OCR
            print("\n=======================================================")
            print(">>> MEMULAI TAHAP 1: DETEKSI KENDARAAN & PLAT NOMOR <<<")
            print("=======================================================")
            raw_df = run_detection(
                video_path=v_path,
                model_path=m_path,
                progress_callback=self.queue_progress
            )
            
            # Tahap 2: Interpolasi
            print("\n=======================================================")
            print(">>> MEMULAI TAHAP 2: INTERPOLASI DATA KOSONG        <<<")
            print("=======================================================")
            self.msg_queue.put(("status", "Melakukan interpolasi lintasan..."))
            interpolated_df = interpolate_missing_data(raw_df)
            
            # Tahap 3: Visualisasi video
            print("\n=======================================================")
            print(">>> MEMULAI TAHAP 3: VISUALISASI HASIL KE VIDEO     <<<")
            print("=======================================================")
            visualize_results(
                video_path=v_path,
                df=interpolated_df,
                output_video_path=out_path,
                progress_callback=self.queue_progress
            )
            
            print("\n=======================================================")
            print(">>> PIPELINE SELESAI DENGAN SUKSES!                 <<<")
            print("=======================================================")
            self.msg_queue.put(("status", "Proses selesai dengan sukses!"))
            self.msg_queue.put(("ui_state", "done"))
            
        except Exception as e:
            print(f"\n[FATAL ERROR]: {e}")
            self.msg_queue.put(("status", f"Gagal - {e}"))
            self.msg_queue.put(("ui_state", "error", str(e)))

    def play_output_video(self):
        out_path = self.output_video_path.get()
        if os.path.exists(out_path):
            try:
                if sys.platform == 'win32':
                    os.startfile(out_path)
                elif sys.platform == 'darwin':
                    import subprocess
                    subprocess.run(['open', out_path])
                else:
                    import subprocess
                    subprocess.run(['xdg-open', out_path])
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memutar video: {e}")
        else:
            messagebox.showerror("Error", "Video hasil tidak ditemukan!")

    # CSV features removed

    def __del__(self):
        # Restore standard streams
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

if __name__ == '__main__':
    root = tk.Tk()
    app = ALPRApp(root)
    root.mainloop()
