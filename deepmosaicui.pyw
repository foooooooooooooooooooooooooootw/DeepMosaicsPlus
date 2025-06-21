#TODO: Image sometimes goes wavy? UI output stops at 2/4 mosaic positions, UI hangs when there are temp files detected and waits for y/n
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import subprocess
import threading
import queue
import time

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class DeepMosaicsUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("DeepMosaicsPlus UI")
        self.root.geometry("800x1000")
        self.root.resizable(True, True)
        
        # Configure grid weight
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.process = None
        
        self.create_widgets()

    def cancel_process(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                # Wait a bit for graceful termination
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()  # Force kill if doesn't terminate
                    
                self.output_text.insert(tk.END, "\n🛑 Process was cancelled by the user.\n")
                self.output_text.see(tk.END)
            except Exception as e:
                self.output_text.insert(tk.END, f"\n❌ Error during cancellation: {e}\n")
                self.output_text.see(tk.END)
            finally:
                self.process = None
        else:
            messagebox.showinfo("Info", "No running process to cancel.")

    def create_widgets(self):
        # Main container with padding
        main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(main_frame, text="DeepMosaics Configuration", 
                                  font=ctk.CTkFont(size=24, weight="bold"))
        title_label.grid(row=0, column=0, pady=(20, 30), sticky="ew")
        
        # Create scrollable frame for all options
        scroll_frame = ctk.CTkScrollableFrame(main_frame, corner_radius=10)
        scroll_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        scroll_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        current_row = 0
        
        # Base Arguments Section
        base_section = ctk.CTkLabel(scroll_frame, text="Base Configuration", 
                                   font=ctk.CTkFont(size=18, weight="bold"))
        base_section.grid(row=current_row, column=0, columnspan=2, pady=(10, 15), sticky="w")
        current_row += 1
        
        # Debug checkbox
        self.debug_var = ctk.BooleanVar()  # Default: checked
        debug_cb = ctk.CTkCheckBox(scroll_frame, text="Debug Mode", variable=self.debug_var)
        debug_cb.grid(row=current_row, column=0, columnspan=2, pady=5, sticky="w")
        current_row += 1
        
        # GPU ID
        gpu_label = ctk.CTkLabel(scroll_frame, text="GPU ID:")
        gpu_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.gpu_entry = ctk.CTkEntry(scroll_frame, placeholder_text="0 (or -1 for CPU)")
        self.gpu_entry.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        self.gpu_entry.insert(0, "0")
        current_row += 1
        
        # Media Path
        media_label = ctk.CTkLabel(scroll_frame, text="Media Path:")
        media_label.grid(row=current_row, column=0, pady=5, sticky="w")
        media_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        media_frame.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        media_frame.grid_columnconfigure(0, weight=1)
        
        self.media_entry = ctk.CTkEntry(media_frame, placeholder_text="Select file or folder...")
        self.media_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        media_btn = ctk.CTkButton(media_frame, text="Browse", width=80, 
                                 command=self.browse_media)
        media_btn.grid(row=0, column=1)
        current_row += 1
        
        # Start Time
        start_label = ctk.CTkLabel(scroll_frame, text="Start Time:")
        start_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.start_entry = ctk.CTkEntry(scroll_frame, placeholder_text="00:00:00")
        self.start_entry.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        self.start_entry.insert(0, "00:00:00")
        current_row += 1
        
        # Duration
        duration_label = ctk.CTkLabel(scroll_frame, text="Duration:")
        duration_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.duration_entry = ctk.CTkEntry(scroll_frame, placeholder_text="00:00:00 (entire video)")
        self.duration_entry.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        self.duration_entry.insert(0, "00:00:00")
        current_row += 1
        
        # Model Path
        model_label = ctk.CTkLabel(scroll_frame, text="Model Path:")
        model_label.grid(row=current_row, column=0, pady=5, sticky="w")
        model_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        model_frame.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        model_frame.grid_columnconfigure(0, weight=1)
        
        self.model_entry = ctk.CTkEntry(model_frame, placeholder_text="Select model file...")
        self.model_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.model_entry.insert(0, "./clean_youknow_video.pth")
        model_btn = ctk.CTkButton(model_frame, text="Browse", width=80,
                                 command=self.browse_model)
        model_btn.grid(row=0, column=1)
        current_row += 1
        
        # Result Directory
        result_label = ctk.CTkLabel(scroll_frame, text="Result Directory:")
        result_label.grid(row=current_row, column=0, pady=5, sticky="w")
        result_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        result_frame.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        result_frame.grid_columnconfigure(0, weight=1)
        
        self.result_entry = ctk.CTkEntry(result_frame, placeholder_text="Output directory...")
        self.result_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.result_entry.insert(0, "./result")
        result_btn = ctk.CTkButton(result_frame, text="Browse", width=80,
                                  command=self.browse_result)
        result_btn.grid(row=0, column=1)
        current_row += 1
        
        # NetG
        netg_label = ctk.CTkLabel(scroll_frame, text="Network G:")
        netg_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.netg_var = ctk.StringVar(value="auto")
        netg_dropdown = ctk.CTkOptionMenu(scroll_frame, variable=self.netg_var,
                                         values=["auto", "unet_128", "unet_256", "resnet_9blocks", "HD", "video"])
        netg_dropdown.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        current_row += 1
        
        # FPS
        fps_label = ctk.CTkLabel(scroll_frame, text="FPS:")
        fps_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.fps_entry = ctk.CTkEntry(scroll_frame, placeholder_text="0 (original)")
        self.fps_entry.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        self.fps_entry.insert(0, "0")
        current_row += 1
        
        # No Preview checkbox
        self.no_preview_var = ctk.BooleanVar()
        no_preview_cb = ctk.CTkCheckBox(scroll_frame, text="No Preview (for server use)", 
                                       variable=self.no_preview_var)
        no_preview_cb.grid(row=current_row, column=0, columnspan=2, pady=5, sticky="w")
        current_row += 1
        
        # Output Size
        output_label = ctk.CTkLabel(scroll_frame, text="Output Size:")
        output_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.output_entry = ctk.CTkEntry(scroll_frame, placeholder_text="0 (original)")
        self.output_entry.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        self.output_entry.insert(0, "0")
        current_row += 1
        
        # Mask Threshold
        mask_label = ctk.CTkLabel(scroll_frame, text="Mask Threshold:")
        mask_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.mask_entry = ctk.CTkEntry(scroll_frame, placeholder_text="48 (0-255)")
        self.mask_entry.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        self.mask_entry.insert(0, "48")
        current_row += 1
        
        # Separator
        separator = ctk.CTkLabel(scroll_frame, text="", height=20)
        separator.grid(row=current_row, column=0, columnspan=2)
        current_row += 1
        
        # Clean Mosaic Section
        clean_section = ctk.CTkLabel(scroll_frame, text="Clean Mosaic Configuration", 
                                    font=ctk.CTkFont(size=18, weight="bold"))
        clean_section.grid(row=current_row, column=0, columnspan=2, pady=(10, 15), sticky="w")
        current_row += 1
        
        # Traditional checkbox - WITH CALLBACK
        self.traditional_var = ctk.BooleanVar()
        traditional_cb = ctk.CTkCheckBox(scroll_frame, text="Use Traditional Method (Not Recommended)", 
                                        variable=self.traditional_var,
                                        command=self.toggle_traditional_fields)  # Added callback
        traditional_cb.grid(row=current_row, column=0, columnspan=2, pady=5, sticky="w")
        current_row += 1
        
        # Traditional Blur - Store reference for enabling/disabling
        self.tr_blur_label = ctk.CTkLabel(scroll_frame, text="Traditional Blur:")
        self.tr_blur_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.tr_blur_entry = ctk.CTkEntry(scroll_frame, placeholder_text="10")
        self.tr_blur_entry.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        self.tr_blur_entry.insert(0, "10")
        current_row += 1
        
        # Traditional Downsample - Store reference for enabling/disabling
        self.tr_down_label = ctk.CTkLabel(scroll_frame, text="Traditional Downsample:")
        self.tr_down_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.tr_down_entry = ctk.CTkEntry(scroll_frame, placeholder_text="10")
        self.tr_down_entry.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        self.tr_down_entry.insert(0, "10")
        current_row += 1
        
        # Initialize traditional fields as disabled
        self.toggle_traditional_fields()
        
        current_row += 1
        
        # No Feather checkbox
        self.no_feather_var = ctk.BooleanVar()
        no_feather_cb = ctk.CTkCheckBox(scroll_frame, text="No Edge Feather (faster)", 
                                       variable=self.no_feather_var)
        no_feather_cb.grid(row=current_row, column=0, columnspan=2, pady=5, sticky="w")
        current_row += 1
        
        # All Mosaic Area checkbox
        self.all_mosaic_var = ctk.BooleanVar(value=False)
        all_mosaic_cb = ctk.CTkCheckBox(scroll_frame, text="Find All Mosaic Areas (Slower)", 
                                       variable=self.all_mosaic_var)
        all_mosaic_cb.grid(row=current_row, column=0, columnspan=2, pady=5, sticky="w")
        current_row += 1
        
        # Median Filter
        medfilt_label = ctk.CTkLabel(scroll_frame, text="Median Filter Window:")
        medfilt_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.medfilt_entry = ctk.CTkEntry(scroll_frame, placeholder_text="5")
        self.medfilt_entry.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        self.medfilt_entry.insert(0, "5")
        current_row += 1
        
        # Expansion Multiplier
        ex_mult_label = ctk.CTkLabel(scroll_frame, text="Expansion Multiplier:")
        ex_mult_label.grid(row=current_row, column=0, pady=5, sticky="w")
        self.ex_mult_entry = ctk.CTkEntry(scroll_frame, placeholder_text="auto")
        self.ex_mult_entry.grid(row=current_row, column=1, pady=5, sticky="ew", padx=(10, 0))
        self.ex_mult_entry.insert(0, "auto")
        current_row += 1
        
        # Action buttons frame
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.grid(row=2, column=0, pady=20, sticky="ew")
        button_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Run button
        run_btn = ctk.CTkButton(button_frame, text="Run", 
                               command=self.run_deepmosaics, height=45,
                               fg_color="#2D5016", hover_color="#3D6B1F",
                               font=ctk.CTkFont(size=16, weight="bold"))
        run_btn.grid(row=0, column=0, padx=10, sticky="ew")
        
        # Clear button
        clear_btn = ctk.CTkButton(button_frame, text="Clear All", 
                                 command=self.clear_all, height=45,
                                 fg_color="#7D2D00", hover_color="#8D3D10",
                                 font=ctk.CTkFont(size=16, weight="bold"))
        clear_btn.grid(row=0, column=1, padx=10, sticky="ew")

        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", 
                           command=self.cancel_process, height=45,
                           fg_color="#7C2020", hover_color="#A83232",
                           font=ctk.CTkFont(size=16, weight="bold"))
        cancel_btn.grid(row=0, column=2, padx=10, sticky="ew")
        
        # Log output area
        output_frame = ctk.CTkFrame(main_frame)
        output_frame.grid(row=3, column=0, pady=(0, 20), sticky="ew")
        output_frame.grid_columnconfigure(0, weight=1)
        
        output_label = ctk.CTkLabel(output_frame, text="Process Log & Errors:", 
                                   font=ctk.CTkFont(size=14, weight="bold"))
        output_label.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        self.output_text = ctk.CTkTextbox(output_frame, height=150, 
                                         font=ctk.CTkFont(family="Consolas", size=12))
        self.output_text.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        # Initial welcome message
        self.output_text.insert("1.0", "Ready to process DeepMosaicsPlus...\nSelect your media and model files, then click 'Run'.\n\n")
    
    def toggle_traditional_fields(self):
        """Enable or disable traditional blur/downsample fields based on checkbox state"""
        if self.traditional_var.get():
            # Enable traditional fields
            self.tr_blur_label.configure(state="normal", text_color=("gray10", "gray90"))
            self.tr_blur_entry.configure(state="normal")
            self.tr_down_label.configure(state="normal", text_color=("gray10", "gray90"))
            self.tr_down_entry.configure(state="normal")
        else:
            # Disable traditional fields
            self.tr_blur_label.configure(state="disabled", text_color=("gray60", "gray40"))
            self.tr_blur_entry.configure(state="disabled")
            self.tr_down_label.configure(state="disabled", text_color=("gray60", "gray40"))
            self.tr_down_entry.configure(state="disabled")
    
    def browse_media(self):
        file_path = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=[("All Files", "*.*"), ("Images", "*.jpg *.jpeg *.png *.bmp"), 
                      ("Videos", "*.mp4 *.avi *.mkv *.mov")]
        )
        if file_path:
            self.media_entry.delete(0, tk.END)
            self.media_entry.insert(0, file_path)
    
    def browse_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, file_path)
    
    def browse_result(self):
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.result_entry.delete(0, tk.END)
            self.result_entry.insert(0, dir_path)
    
    def generate_command(self):
        cmd = ["python", "deepmosaic.py"]
        
        # Base arguments
        if self.debug_var.get():
            cmd.append("--debug")
        
        gpu_id = self.gpu_entry.get().strip()
        if gpu_id:
            cmd.extend(["--gpu_id", gpu_id])
        
        media_path = self.media_entry.get().strip()
        if media_path:
            cmd.extend(["--media_path", media_path])
        
        start_time = self.start_entry.get().strip()
        if start_time and start_time != "00:00:00":
            cmd.extend(["-ss", start_time])
        
        duration = self.duration_entry.get().strip()
        if duration and duration != "00:00:00":
            cmd.extend(["-t", duration])
        
        model_path = self.model_entry.get().strip()
        if model_path:
            cmd.extend(["--model_path", model_path])
        
        result_dir = self.result_entry.get().strip()
        if result_dir:
            cmd.extend(["--result_dir", result_dir])
        
        netg = self.netg_var.get()
        if netg != "auto":
            cmd.extend(["--netG", netg])
        
        fps = self.fps_entry.get().strip()
        if fps and fps != "0":
            cmd.extend(["--fps", fps])
        
        if self.no_preview_var.get():
            cmd.append("--no_preview")
        
        output_size = self.output_entry.get().strip()
        if output_size and output_size != "0":
            cmd.extend(["--output_size", output_size])
        
        mask_threshold = self.mask_entry.get().strip()
        if mask_threshold and mask_threshold != "48":
            cmd.extend(["--mask_threshold", mask_threshold])
        
        # Clean mosaic arguments
        if self.traditional_var.get():
            cmd.append("--traditional")
            
            # Only add traditional values if the checkbox is enabled
            tr_blur = self.tr_blur_entry.get().strip()
            if tr_blur and tr_blur != "10":
                cmd.extend(["--tr_blur", tr_blur])
            
            tr_down = self.tr_down_entry.get().strip()
            if tr_down and tr_down != "10":
                cmd.extend(["--tr_down", tr_down])
        
        if self.no_feather_var.get():
            cmd.append("--no_feather")
        
        if self.all_mosaic_var.get():
            cmd.append("--all_mosaic_area")
        
        medfilt = self.medfilt_entry.get().strip()
        if medfilt and medfilt != "5":
            cmd.extend(["--medfilt_num", medfilt])
        
        ex_mult = self.ex_mult_entry.get().strip()
        if ex_mult and ex_mult != "auto":
            cmd.extend(["--ex_mult", ex_mult])
        
        return cmd
    
    def update_output_text(self, text):
        """Thread-safe method to update output text"""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
    
    def run_deepmosaics(self):
        def run_in_thread():
            try:
                cmd = self.generate_command()

                # Clear previous output and show command being run
                self.output_text.delete("1.0", tk.END)
                command_str = " ".join(cmd)
                self.output_text.insert("1.0", f"🚀 Running: {command_str}\n")
                self.output_text.insert(tk.END, "=" * 60 + "\n\n")

                script_dir = os.path.dirname(os.path.abspath(__file__))

                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'

                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                    text=True,
                    universal_newlines=True,
                    cwd=script_dir,
                    env=env
                )

                output_queue = queue.Queue()
                prompt_event = threading.Event()

                def read_output():
                    try:
                        for line in iter(self.process.stdout.readline, ''):
                            if line:
                                line_lower = line.lower().strip()
                                if ('continue' in line_lower and 'y/n' in line_lower) or \
                                ('unfinished' in line_lower and 'continue' in line_lower and '?' in line_lower):
                                    output_queue.put(('prompt', line))
                                    prompt_event.set()
                                else:
                                    output_queue.put(('output', line))
                    except:
                        pass
                    output_queue.put(('done', None))

                output_thread = threading.Thread(target=read_output)
                output_thread.daemon = True
                output_thread.start()

                time.sleep(0.1)  # Let the subprocess emit output early

                while True:
                    try:
                        msg_type, content = output_queue.get(timeout=0.1)

                        if msg_type == 'done':
                            break
                        elif msg_type == 'prompt':
                            self.root.after(0, lambda text=content: self.update_output_text(text))
                            self.root.update()

                            result = messagebox.askyesno(
                                "Continue Processing",
                                "An unfinished process was detected. Do you want to continue it?",
                                default='yes'
                            )

                            try:
                                if self.process and self.process.poll() is None:
                                    if result:
                                        self.process.stdin.write('y\n')
                                        response_text = "✅ User selected: Yes (continue)\n"
                                    else:
                                        self.process.stdin.write('n\n')
                                        response_text = "❌ User selected: No (abort)\n"
                                    self.process.stdin.flush()
                                    self.root.after(0, lambda text=response_text: self.update_output_text(text))
                            except Exception as e:
                                error_text = f"❌ Error sending response: {e}\n"
                                self.root.after(0, lambda text=error_text: self.update_output_text(text))

                            prompt_event.clear()

                        elif msg_type == 'output':
                            self.root.after(0, lambda text=content: self.update_output_text(text))

                    except queue.Empty:
                        self.root.update_idletasks()
                        if self.process and self.process.poll() is not None:
                            while True:
                                try:
                                    msg_type, content = output_queue.get_nowait()
                                    if msg_type in ['output', 'prompt']:
                                        self.root.after(0, lambda text=content: self.update_output_text(text))
                                except queue.Empty:
                                    break
                            break

                if self.process:
                    self.process.wait()
                    return_code = self.process.returncode
                    self.process = None
                else:
                    return_code = -1

                def update_final_status():
                    self.output_text.insert(tk.END, "\n" + "=" * 60 + "\n")
                    if return_code == 0:
                        self.output_text.insert(tk.END, "✅ Process completed successfully!\n")
                    else:
                        self.output_text.insert(tk.END, f"❌ Process failed with return code: {return_code}\n")
                    self.output_text.see(tk.END)

                self.root.after(0, update_final_status)

            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.output_text.insert(tk.END, f"\n❌ Error: {msg}\n"))

        if not self.media_entry.get().strip():
            messagebox.showerror("Error", "Please select a media file!")
            return

        if not self.model_entry.get().strip():
            messagebox.showerror("Error", "Please select a model file!")
            return

        thread = threading.Thread(target=run_in_thread)
        thread.daemon = True
        thread.start()


    def clear_all(self):
        # Reset all fields to default values
        self.debug_var.set(False)
        self.gpu_entry.delete(0, tk.END)
        self.gpu_entry.insert(0, "0")
        self.media_entry.delete(0, tk.END)
        self.start_entry.delete(0, tk.END)
        self.start_entry.insert(0, "00:00:00")
        self.duration_entry.delete(0, tk.END)
        self.duration_entry.insert(0, "00:00:00")
        self.model_entry.delete(0, tk.END)
        self.model_entry.insert(0, "./clean_youknow_video.pth")
        self.result_entry.delete(0, tk.END)
        self.result_entry.insert(0, "./result")
        self.netg_var.set("auto")
        self.fps_entry.delete(0, tk.END)
        self.fps_entry.insert(0, "0")
        self.no_preview_var.set(False)
        self.output_entry.delete(0, tk.END)
        self.output_entry.insert(0, "0")
        self.mask_entry.delete(0, tk.END)
        self.mask_entry.insert(0, "48")
        
        self.traditional_var.set(False)
        self.tr_blur_entry.delete(0, tk.END)
        self.tr_blur_entry.insert(0, "10")
        self.tr_down_entry.delete(0, tk.END)
        self.tr_down_entry.insert(0, "10")
        self.no_feather_var.set(False)
        self.all_mosaic_var.set(True)  # This was False in original, but you set it to True by default
        self.medfilt_entry.delete(0, tk.END)
        self.medfilt_entry.insert(0, "5")
        self.ex_mult_entry.delete(0, tk.END)
        self.ex_mult_entry.insert(0, "auto")
        
        # Update traditional fields state after clearing
        self.toggle_traditional_fields()
        
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", "Ready to process DeepMosaicsPlus...\nSelect your media and model files, then click 'Run'.\n\n")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DeepMosaicsUI()
    app.run()