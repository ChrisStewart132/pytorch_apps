import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
import abc
import subprocess
import os
import logging
import threading
import queue

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class AbstractTab(abc.ABC):
    def __init__(self, master, tab_name):
        super().__init__()
        self.master = master
        self.tab_name = tab_name
        self.frame = ttk.Frame(master)
        self.subprocess_process = None  # To store the subprocess object for killing

        self.output_text = tk.Text(self.frame, height=10, width=50)
        self.output_text.grid(row=10, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        self.output_scrollbar = ttk.Scrollbar(self.frame, orient='vertical', command=self.output_text.yview)
        self.output_scrollbar.grid(row=10, column=2, sticky='ns')
        self.output_text.config(yscrollcommand=self.output_scrollbar.set)

        self.kill_button = ttk.Button(self.frame, text="Kill Process", command=self.kill_current_subprocess)
        self.kill_button.grid(row=11, column=0, columnspan=2, pady=5)

        self.frame.rowconfigure(10, weight=1)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.command_queue = queue.Queue() # Initialize command_queue here

    @abc.abstractmethod
    def create_content(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass

    def run_subprocess(self, command):
        self.output_text.insert(tk.END, f"Running command: {' '.join(command)}\n") # Indicate command start
        self.output_text.see(tk.END)
        self.output_text.update_idletasks()
        self.kill_button.config(state=tk.NORMAL) # Enable Kill Process button when process starts

        # Clear previous queue if any, although it should be empty each time
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break

        thread = threading.Thread(target=self._subprocess_thread, args=(command,))
        thread.daemon = True # Allow main thread to exit even if thread is running
        thread.start()
        self.master.after(100, self._check_subprocess_queue) # Start checking queue

    def _subprocess_thread(self, command): # 'command' is already a list in your run_subprocess methods
        try:
            # Remove shell=True and pass command as a list
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
                self.subprocess_process = process # Store process object
                for output in process.stdout:
                    self.command_queue.put(output) # Put stdout into queue

                stderr_output = process.stderr.read()
                if stderr_output:
                    self.command_queue.put(stderr_output) # Put stderr into queue
        except FileNotFoundError:
            self.command_queue.put(f"Error: Command not found: {command[0]}\n") # Access command name from list
        except Exception as e:
            self.command_queue.put(f"An error occurred: {e}\n")
            logging.error(f"Subprocess error: {e}", exc_info=True)
        finally:
            self.command_queue.put(None) # Signal thread completion
            self.subprocess_process = None # Reset process object when thread finishes
            self.master.after(0, self.disable_kill_button) # Disable kill button in main thread

    def _check_subprocess_queue(self):
        try:
            while True:
                output = self.command_queue.get_nowait() # Non-blocking queue get
                if output is None: # Thread finished signal
                    break
                self.output_text.insert(tk.END, output)
                self.output_text.see(tk.END)
                self.output_text.update_idletasks()
        except queue.Empty: # Queue is empty, check again later
            pass
        self.master.after(100, self._check_subprocess_queue) # Check queue again in 100ms

    def kill_current_subprocess(self):
        if self.subprocess_process and self.subprocess_process.poll() is None: # Check if process is running
            self.subprocess_process.terminate() # Send SIGTERM
            self.output_text.insert(tk.END, "\n[Process killed by user]\n")
            self.output_text.see(tk.END)
            self.output_text.update_idletasks()
            self.subprocess_process = None # Reset process object
            self.disable_kill_button()
        else:
            self.output_text.insert(tk.END, "\n[No process to kill]\n")
            self.output_text.see(tk.END)
            self.output_text.update_idletasks()

    def disable_kill_button(self):
        self.kill_button.config(state=tk.DISABLED)


    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class TrainingTab(AbstractTab):
    def __init__(self, master):
        super().__init__(master, "Training")
        self.create_content()

    def create_content(self):
        # Number of Features
        n_features_label = ttk.Label(self.frame, text="Number of Features:")
        n_features_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.n_features_spinbox = tk.Spinbox(self.frame, from_=1, to=1000, increment=1, width=40)
        self.n_features_spinbox.delete(0, tk.END)
        self.n_features_spinbox.insert(0, "16")
        self.n_features_spinbox.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Starting Epoch
        starting_epoch_label = ttk.Label(self.frame, text="Starting Epoch:")
        starting_epoch_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.starting_epoch_spinbox = tk.Spinbox(self.frame, from_=0, to=1000, increment=1, width=40)
        self.starting_epoch_spinbox.delete(0, tk.END)
        self.starting_epoch_spinbox.insert(0, "0")
        self.starting_epoch_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Number of Epochs
        epochs_label = ttk.Label(self.frame, text="Epochs:")
        epochs_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.epochs_spinbox = tk.Spinbox(self.frame, from_=1, to=10000, increment=1, width=40)
        self.epochs_spinbox.delete(0, tk.END)
        self.epochs_spinbox.insert(0, "12")
        self.epochs_spinbox.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Learning Rate
        learning_rate_label = ttk.Label(self.frame, text="Learning Rate:")
        learning_rate_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.learning_rate_spinbox = tk.Spinbox(self.frame, from_=0.00001, to=1.0, increment=0.0001, format="%.5f", width=40)
        self.learning_rate_spinbox.delete(0, tk.END)
        self.learning_rate_spinbox.insert(0, "0.001")
        self.learning_rate_spinbox.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Weight Decay
        weight_decay_label = ttk.Label(self.frame, text="Weight Decay:")
        weight_decay_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.weight_decay_spinbox = tk.Spinbox(self.frame, from_=0.0, to=1.0, increment=0.01, format="%.2f", width=40)
        self.weight_decay_spinbox.delete(0, tk.END)
        self.weight_decay_spinbox.insert(0, "0.01")
        self.weight_decay_spinbox.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Batch Size
        batch_size_label = ttk.Label(self.frame, text="Batch Size:")
        batch_size_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.batch_size_spinbox = tk.Spinbox(self.frame, from_=1, to=128, increment=1, width=40)
        self.batch_size_spinbox.delete(0, tk.END)
        self.batch_size_spinbox.insert(0, "8")
        self.batch_size_spinbox.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # Scale Factor
        scale_factor_label = ttk.Label(self.frame, text="Scale Factor:")
        scale_factor_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.scale_factor_spinbox = tk.Spinbox(self.frame, from_=1, to=8, increment=1, width=40)
        self.scale_factor_spinbox.delete(0, tk.END)
        self.scale_factor_spinbox.insert(0, "2")
        self.scale_factor_spinbox.grid(row=6, column=1, padx=5, pady=5, sticky="w")

        train_button = ttk.Button(self.frame, text="Start Training", command=self.start_training)
        train_button.grid(row=7, column=0, columnspan=2, pady=10)

    def start_training(self):
        n_features_str = self.n_features_spinbox.get()
        starting_epoch_str = self.starting_epoch_spinbox.get()
        epochs_str = self.epochs_spinbox.get()
        learning_rate_str = self.learning_rate_spinbox.get()
        weight_decay_str = self.weight_decay_spinbox.get()
        batch_size_str = self.batch_size_spinbox.get()
        scale_factor_str = self.scale_factor_spinbox.get()

        try:
            n_features = int(n_features_str)
            starting_epoch = int(starting_epoch_str)
            epochs = int(epochs_str)
            learning_rate = float(learning_rate_str)
            weight_decay = float(weight_decay_str)
            batch_size = int(batch_size_str)
            scale_factor = int(scale_factor_str)

            if learning_rate <= 0 or epochs <= 0 or n_features <= 0 or batch_size <= 0 or scale_factor <= 0:
                messagebox.showerror("Input Error", "All relevant values must be positive.")
                return
            if weight_decay < 0 :
                messagebox.showerror("Input Error", "Weight decay must be non-negative.")
                return
            if starting_epoch < 0:
                messagebox.showerror("Input Error", "Starting epoch must be non-negative.")
                return


            command = [
                "python",
                "train_cli.py",
                "--n_features", str(n_features),
                "--starting_epoch", str(starting_epoch),
                "--n_epochs", str(epochs),
                "--learning_rate", str(learning_rate),
                "--weight_decay", str(weight_decay),
                "--batch_size", str(batch_size),
                "--scale_factor", str(scale_factor)
            ]
            self.run_subprocess(command)

        except ValueError:
            messagebox.showerror("Input Error", "Invalid input. Please enter numeric values for all fields.")

    def update(self):
        pass


class InferenceTab(AbstractTab):
    def __init__(self, master, root_dir):
        super().__init__(master, "Inference")
        self.root_dir = root_dir
        self.models = self.find_models(root_dir)
        self.videos = self.find_videos()
        self.create_content()

    def find_models(self, root_dir):
        models = []
        search_dir = root_dir
        try:
            for root, _, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.pth'):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, root_dir)
                        models.append(relative_path)
        except OSError as e:
            messagebox.showerror("Error", f"Could not access model directory: {search_dir}\n{e}")
            logging.error(f"Error accessing model directory: {search_dir}", exc_info=True)
            return []
        return models

    def find_videos(self):
        videos = []
        search_dir = self.root_dir
        try:
            for root, _, files in os.walk(search_dir):
                for file in files:
                    if file.endswith(('.avi', '.mp4')):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, self.root_dir)
                        videos.append(relative_path)
        except OSError as e:
            messagebox.showerror("Error", f"Could not access video directory: {search_dir}\n{e}")
            logging.error(f"Error accessing video directory: {search_dir}", exc_info=True)
            return []
        return videos

    def create_content(self):
        self.model_var = tk.StringVar(self.frame)
        self.video_var = tk.StringVar(self.frame)
        self.save_video_var = tk.BooleanVar(self.frame, value=False)

        display_models = self.models # Relative paths for models
        display_videos = self.videos # Full relative paths for videos

        if display_models:
            self.model_var.set(display_models[0])
        if display_videos:
            self.video_var.set(display_videos[0])

        model_label = ttk.Label(self.frame, text="Select Model:")
        model_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.model_dropdown = ttk.Combobox(self.frame, textvariable=self.model_var, values=display_models, width=60) # Wider
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.model_dropdown.config(justify='right') # Right justify

        video_label = ttk.Label(self.frame, text="Select Video:")
        video_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.video_dropdown = ttk.Combobox(self.frame, textvariable=self.video_var, values=display_videos, width=60) # Wider
        self.video_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.video_dropdown.config(justify='right') # Right justify


        self.save_video_checkbox = tk.Checkbutton(self.frame, text="Save Video", variable=self.save_video_var)
        self.save_video_checkbox.grid(row=2, column=0, columnspan=2, pady=5)

        self.test_button = tk.Button(self.frame, text="Run Inference", command=self.run_command)
        self.test_button.grid(row=3, column=0, columnspan=2, pady=20)

        self.refresh_button = ttk.Button(self.frame, text="Refresh Files", command=self.update_files)
        self.refresh_button.grid(row=4, column=0, columnspan=2, pady=10)

    def run_command(self):
        model = self.model_var.get()
        video = self.video_var.get()
        save_video = self.save_video_var.get()

        if not model or not video:
            messagebox.showerror("Error", "Please select both a model and a video!")
            return

        full_model_path = os.path.join(self.root_dir, model) # Reconstruct full model path
        full_video_path = os.path.join(self.root_dir, video)

        command_list = [
            "python",
            "infer_cli.py",
            "--model_path", full_model_path,
            "--video_path", full_video_path,
        ]

        if save_video:
            command_list.append("--save_video")

        print("Executing command:", command_list)
        self.run_subprocess(command_list)

    def update(self):
        pass

    def update_files(self):
        self.models = self.find_models(self.root_dir)
        self.videos = self.find_videos()
        self.model_dropdown['values'] = self.models
        self.video_dropdown['values'] = self.videos
        if self.models:
            self.model_var.set(self.models[0])
            self.model_dropdown.current(0)
        if self.videos:
            self.video_var.set(self.videos[0])
            self.video_dropdown.current(0)
        messagebox.showinfo("Info", "File lists refreshed.")


class DataTab(AbstractTab):
    def __init__(self, master, root_dir):
        super().__init__(master, "Data")
        self.root_dir = root_dir
        self.validation_checkbox = None # Placeholder for validation checkbox
        self.test_checkbox = None       # Placeholder for test checkbox
        self.train_checkbox = None      # Placeholder for train checkbox
        self.frames_skipped_spinbox = None # Placeholder for frames_skipped spinbox

        # Flag Variables (no longer associated with Treeview items directly)
        self.validation_var = tk.BooleanVar(value=False)
        self.test_var = tk.BooleanVar(value=False)
        self.train_var = tk.BooleanVar(value=False)

        self.create_content()

    def create_content(self):
        # Flags and Videos to Images Frame (placed directly in self.frame grid now)
        v_to_i_frame = ttk.Frame(self.frame)
        v_to_i_frame.grid(row=0, column=0, columnspan=3, pady=10, padx=5, sticky='ew') # Row 0 now

        # Flag Checkboxes
        flags_frame = ttk.Frame(v_to_i_frame)
        flags_frame.pack(side=tk.LEFT, padx=10)

        self.validation_checkbox = tk.Checkbutton(flags_frame, text="Validation", variable=self.validation_var) # Use self.validation_var
        self.validation_checkbox.pack(side=tk.TOP, anchor='w')
        self.test_checkbox = tk.Checkbutton(flags_frame, text="Test", variable=self.test_var) # Use self.test_var
        self.test_checkbox.pack(side=tk.TOP, anchor='w')
        self.train_checkbox = tk.Checkbutton(flags_frame, text="Train", variable=self.train_var) # Use self.train_var
        self.train_checkbox.pack(side=tk.TOP, anchor='w')

        # Videos to Images Controls
        v_to_i_controls_frame = ttk.Frame(v_to_i_frame)
        v_to_i_controls_frame.pack(side=tk.LEFT, padx=20)

        frames_skipped_label = ttk.Label(v_to_i_controls_frame, text="Frames Skipped:")
        frames_skipped_label.pack(side=tk.TOP, anchor='w')
        self.frames_skipped_spinbox = tk.Spinbox(v_to_i_controls_frame, from_=1, to=1000, increment=1, width=10)
        self.frames_skipped_spinbox.delete(0, tk.END)
        self.frames_skipped_spinbox.insert(0, "1") # Default value
        self.frames_skipped_spinbox.pack(side=tk.TOP, anchor='w')

        self.videos_to_images_button = ttk.Button(v_to_i_controls_frame, text="Videos to Images", command=self.videos_to_images_command)
        self.videos_to_images_button.pack(side=tk.TOP, pady=5, anchor='w') # Below frames_skipped


        self.frame.rowconfigure(0, weight=0) # Adjust row weight if needed, set to 0 if no longer expanding
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.frame.columnconfigure(2, weight=1)


    def videos_to_images_command(self):
        frames_skipped = self.frames_skipped_spinbox.get()
        video_dir = self.root_dir # Or define a specific video directory input field if needed
        output_dir = os.path.join(self.root_dir, "images_output") # Example output directory

        
        """
        extract frames from videos
        usage:
            python video_to_images_cli.py --video_dir <video_dir> --output_dir_for_images <output_dir> --frames_skipped <frames_skipped>

        python video_to_images_cli.py --video_dir train/input --output_dir train/input 
        python video_to_images_cli.py --video_dir train/output --output_dir train/output 

        python video_to_images_cli.py --video_dir validation/input --output_dir validation/input 
        python video_to_images_cli.py --video_dir validation/output --output_dir validation/output 

        python video_to_images_cli.py --video_dir test/input --output_dir test/input 
        python video_to_images_cli.py --video_dir test/output --output_dir test/output 
        """
    def videos_to_images_command(self):
        frames_skipped = self.frames_skipped_spinbox.get()
        data_dir = "data"
        video_dir_train_input = os.path.join(self.root_dir, data_dir, "train", "input")
        output_dir_train_input = os.path.join(self.root_dir, data_dir, "train", "input")

        video_dir_train_output = os.path.join(self.root_dir, data_dir, "train", "output")
        output_dir_train_output = os.path.join(self.root_dir, data_dir, "train", "output")

        video_dir_test_input = os.path.join(self.root_dir, data_dir, "test", "input")
        output_dir_test_input = os.path.join(self.root_dir, data_dir, "test", "input")

        video_dir_test_output = os.path.join(self.root_dir, data_dir, "test", "output")
        output_dir_test_output = os.path.join(self.root_dir, data_dir, "test", "output")

        video_dir_validation_input = os.path.join(self.root_dir, data_dir, "validation", "input")
        output_dir_validation_input = os.path.join(self.root_dir, data_dir, "validation", "input")

        video_dir_validation_output = os.path.join(self.root_dir, data_dir, "validation", "output")
        output_dir_validation_output = os.path.join(self.root_dir, data_dir, "validation", "output")

        if self.validation_var.get():
            # --- Command for validation/input ---
            if os.path.exists(video_dir_validation_input): # Check if directory exists
                command_validation_input = [
                    "python",
                    os.path.join("data", "videos_to_images_cli.py"),
                    "--video_dir", video_dir_validation_input,
                    "--output_dir", output_dir_validation_input,
                    "--frames_skipped", str(frames_skipped)
                ]
                self.run_subprocess(command_validation_input)
            else:
                error_message = f"Error: Validation input video directory not found: {video_dir_validation_input}"
                self.output_text.insert(tk.END, "\n" + error_message + "\n")
                self.output_text.see(tk.END)
                messagebox.showerror("Directory Error", error_message) # Optionally show error popup

            # --- Command for validation/output ---
            if os.path.exists(video_dir_validation_output): # Check if directory exists
                command_validation_output = [
                    "python",
                    os.path.join("data", "videos_to_images_cli.py"),
                    "--video_dir", video_dir_validation_output,
                    "--output_dir", output_dir_validation_output,
                    "--frames_skipped", str(frames_skipped)
                ]
                self.run_subprocess(command_validation_output)
            else:
                error_message = f"Error: Validation output video directory not found: {video_dir_validation_output}"
                self.output_text.insert(tk.END, "\n" + error_message + "\n")
                self.output_text.see(tk.END)
                messagebox.showerror("Directory Error", error_message) # Optionally show error popup


        if self.test_var.get():
            # --- Command for test/input ---
            if os.path.exists(video_dir_test_input): # Check if directory exists
                command_test_input = [
                    "python",
                    os.path.join("data", "videos_to_images_cli.py"),
                    "--video_dir", video_dir_test_input,
                    "--output_dir", output_dir_test_input,
                    "--frames_skipped", str(frames_skipped)
                ]
                self.run_subprocess(command_test_input)
            else:
                error_message = f"Error: Test input video directory not found: {video_dir_test_input}"
                self.output_text.insert(tk.END, "\n" + error_message + "\n")
                self.output_text.see(tk.END)
                messagebox.showerror("Directory Error", error_message) # Optionally show error popup


            # --- Command for test/output ---
            if os.path.exists(video_dir_test_output): # Check if directory exists
                command_test_output = [
                    "python",
                    os.path.join("data", "videos_to_images_cli.py"),
                    "--video_dir", video_dir_test_output,
                    "--output_dir", output_dir_test_output,
                    "--frames_skipped", str(frames_skipped)
                ]
                self.run_subprocess(command_test_output)
            else:
                error_message = f"Error: Test output video directory not found: {video_dir_test_output}"
                self.output_text.insert(tk.END, "\n" + error_message + "\n")
                self.output_text.see(tk.END)
                messagebox.showerror("Directory Error", error_message) # Optionally show error popup

        if self.train_var.get():
            # --- Command for train/input ---
            if os.path.exists(video_dir_train_input): # Check if directory exists
                command_train_input = [
                    "python",
                    os.path.join("data", "videos_to_images_cli.py"),
                    "--video_dir", video_dir_train_input,
                    "--output_dir", output_dir_train_input,
                    "--frames_skipped", str(frames_skipped)
                ]
                self.run_subprocess(command_train_input)
            else:
                error_message = f"Error: Train input video directory not found: {video_dir_train_input}"
                self.output_text.insert(tk.END, "\n" + error_message + "\n")
                self.output_text.see(tk.END)
                messagebox.showerror("Directory Error", error_message) # Optionally show error popup

            # --- Command for train/output ---
            if os.path.exists(video_dir_train_output): # Check if directory exists
                command_train_output = [
                    "python",
                    os.path.join("data", "videos_to_images_cli.py"),
                    "--video_dir", video_dir_train_output,
                    "--output_dir", output_dir_train_output,
                    "--frames_skipped", str(frames_skipped)
                ]
                self.run_subprocess(command_train_output)
            else:
                error_message = f"Error: Train output video directory not found: {video_dir_train_output}"
                self.output_text.insert(tk.END, "\n" + error_message + "\n")
                self.output_text.see(tk.END)
                messagebox.showerror("Directory Error", error_message) # Optionally show error popup


    def update(self):
        # No Treeview to update anymore
        pass


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.root_dir = os.getcwd()
        self.title("GUI")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")

        self.inference_tab = InferenceTab(self.notebook, self.root_dir)
        self.notebook.add(self.inference_tab.frame, text=self.inference_tab.tab_name)

        self.train_tab = TrainingTab(self.notebook)
        self.notebook.add(self.train_tab.frame, text=self.train_tab.tab_name)

        self.data_tab = DataTab(self.notebook, self.root_dir) # Create DataTab instance
        self.notebook.add(self.data_tab.frame, text=self.data_tab.tab_name) # Add DataTab to notebook


        self.mainloop()


    def find_files(self):
        pass


if __name__ == "__main__":
    app = MainWindow()