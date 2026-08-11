"""
WiFi Setup Tab for the security system GUI.

Drop this into your project alongside multi_camera.py and add it as a
tab/frame in your existing Tkinter Notebook. Requires Raspberry Pi OS
with NetworkManager (default on current Raspberry Pi OS).

Usage inside your main app (example):

    from wifi_tab import WiFiTab
    notebook = ttk.Notebook(root)
    wifi_tab = WiFiTab(notebook)
    notebook.add(wifi_tab, text="WiFi Setup")

PERMISSIONS NOTE:
`nmcli` network connect/scan operations normally require root, or a
polkit rule granting your user passwordless access. If you hit
permission errors, either run the app with sudo (quick, not ideal)
or set up a polkit rule (ask me for this and I'll generate it).
"""

import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox


class WiFiTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._build_ui()
        self.refresh_networks()
        self.refresh_status()

    def _build_ui(self):
        # Status row
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", padx=10, pady=(10, 5))

        ttk.Label(status_frame, text="Connected to:").pack(side="left")
        self.status_label = ttk.Label(status_frame, text="Checking...", font=("", 10, "bold"))
        self.status_label.pack(side="left", padx=(5, 0))

        # Network list
        list_frame = ttk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Label(list_frame, text="Available Networks:").pack(anchor="w")

        self.network_listbox = tk.Listbox(list_frame, height=8)
        self.network_listbox.pack(fill="both", expand=True, pady=(5, 0))

        refresh_btn = ttk.Button(list_frame, text="Refresh", command=self.refresh_networks)
        refresh_btn.pack(anchor="e", pady=(5, 0))

        # Password + connect
        connect_frame = ttk.Frame(self)
        connect_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(connect_frame, text="Password:").pack(side="left")
        self.password_var = tk.StringVar()
        password_entry = ttk.Entry(connect_frame, textvariable=self.password_var, show="*")
        password_entry.pack(side="left", fill="x", expand=True, padx=5)

        self.connect_btn = ttk.Button(connect_frame, text="Connect", command=self._on_connect_clicked)
        self.connect_btn.pack(side="left")

        # Inline feedback
        self.feedback_label = ttk.Label(self, text="", foreground="red")
        self.feedback_label.pack(padx=10, anchor="w")

    def refresh_status(self):
        def worker():
            try:
                result = subprocess.run(
                    ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
                    capture_output=True, text=True, timeout=10
                )
                current = None
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("yes:"):
                        current = line.split(":", 1)[1]
                        break
                self.after(0, lambda: self.status_label.config(
                    text=current if current else "Not connected"
                ))
            except Exception as e:
                self.after(0, lambda: self.status_label.config(text="Unknown"))
        threading.Thread(target=worker, daemon=True).start()

    def refresh_networks(self):
        self.network_listbox.delete(0, tk.END)
        self.network_listbox.insert(tk.END, "Scanning...")

        def worker():
            try:
                subprocess.run(["nmcli", "dev", "wifi", "rescan"], timeout=10)
                result = subprocess.run(
                    ["nmcli", "-t", "-f", "SSID,SIGNAL", "dev", "wifi", "list"],
                    capture_output=True, text=True, timeout=10
                )
                seen = set()
                networks = []
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.rsplit(":", 1)
                    if len(parts) != 2:
                        continue
                    ssid, signal = parts
                    if ssid and ssid not in seen:
                        seen.add(ssid)
                        networks.append((ssid, signal))
                networks.sort(key=lambda n: int(n[1]) if n[1].isdigit() else 0, reverse=True)

                def update_list():
                    self.network_listbox.delete(0, tk.END)
                    for ssid, signal in networks:
                        self.network_listbox.insert(tk.END, f"{ssid}  ({signal}%)")
                self.after(0, update_list)
            except Exception as e:
                self.after(0, lambda: self._show_feedback(f"Scan failed: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_connect_clicked(self):
        selection = self.network_listbox.curselection()
        if not selection:
            self._show_feedback("Select a network first.")
            return

        raw = self.network_listbox.get(selection[0])
        ssid = raw.rsplit("  (", 1)[0]
        password = self.password_var.get()

        if not password:
            self._show_feedback("Enter a password.")
            return

        self.connect_btn.config(state="disabled", text="Connecting...")
        self._show_feedback("")

        def worker():
            try:
                result = subprocess.run(
                    ["nmcli", "dev", "wifi", "connect", ssid, "password", password],
                    capture_output=True, text=True, timeout=30
                )
                success = result.returncode == 0

                def finish():
                    self.connect_btn.config(state="normal", text="Connect")
                    if success:
                        messagebox.showinfo("Connected", f"Successfully connected to {ssid}")
                        self.refresh_status()
                    else:
                        self._show_feedback(result.stderr.strip() or "Connection failed.")
                self.after(0, finish)
            except Exception as e:
                def finish():
                    self.connect_btn.config(state="normal", text="Connect")
                    self._show_feedback(f"Error: {e}")
                self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _show_feedback(self, message):
        self.feedback_label.config(text=message)


# Standalone test — run this file directly to preview the tab in isolation
if __name__ == "__main__":
    root = tk.Tk()
    root.title("WiFi Tab Preview")
    root.geometry("400x400")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)
    tab = WiFiTab(notebook)
    notebook.add(tab, text="WiFi Setup")
    root.mainloop()
