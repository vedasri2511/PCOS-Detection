import tkinter as tk
from tkinter import ttk
import requests
import serial

# === GSR READER ===
def get_skin_conductance(port='COM3', baudrate=115200):
    try:
        with serial.Serial(port, baudrate, timeout=2) as ser:
            for _ in range(15):  # Skip initial ESP32 boot messages
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print("[DEBUG] Serial Line:", line)
                try:
                    voltage = float(line)
                    if 0.1 < voltage < 3.3:
                        return voltage
                except ValueError:
                    continue
    except Exception as e:
        print("Serial read error:", e)
    return None

# === GUI Submission Functions ===
def submit_clinical_data():
    try:
        skin_conductance = get_skin_conductance()
        if skin_conductance is None:
            prediction_label.config(text="Error: No valid GSR data received.")
            return

        fsh = float(fsh_entry.get())
        lh = float(lh_entry.get())
        amh = float(amh_entry.get())
        fsh_lh = fsh / lh if lh != 0 else 0
        tsh = float(tsh_entry.get())
        prl = float(prl_entry.get())
        vit_d3 = float(vit_d3_entry.get())

        clinical = {
            "FSH": fsh, "LH": lh, "AMH": amh,
            "FSH_LH": fsh_lh, "TSH": tsh, "PRL": prl, "VitD3": vit_d3
        }

        gsr_features = {
            "SkinConductance": skin_conductance
        }

        payload = {
            "gsr_features": gsr_features,
            "clinical": clinical,
            "non_clinical": {}
        }

        print("[Sending Clinical Payload]", payload)

        response = requests.post("http://127.0.0.1:5000/predict_all", json=payload)
        if response.status_code == 200:
            result = response.json()
            prediction_label.config(
                text=f"Stress: {result['stress_level']} | PCOS Risk: {result['pcos_risk']} ({result['pcos_probability']}%)"
            )
        else:
            prediction_label.config(text="Error: Couldn't get prediction.")
    except Exception as e:
        print("[GUI ERROR]", e)
        import traceback
        traceback.print_exc()
        prediction_label.config(text=f"Error: {str(e)}")

def submit_non_clinical_data():
    try:
        skin_conductance = get_skin_conductance()
        if skin_conductance is None:
            prediction_label.config(text="Error: No valid GSR data received.")
            return

        age = int(age_entry.get())
        weight = float(weight_entry.get())
        height = float(height_entry.get())
        bmi = float(bmi_entry.get())
        cycle_length = int(cycle_length_entry.get())

        non_clinical = {
            "AGE": age,
            "WeightKg": weight,
            "HeightCm": height,
            "BMI": bmi,
            "CYCLE_LENGTH": cycle_length,
            "WeightGain": weight_gain_var.get(),
            "HairGrowth": hair_growth_var.get(),
            "SkinDarkening": skin_darkening_var.get(),
            "HairLoss": hair_loss_var.get(),
            "FastFood": fast_food_var.get(),
            "RegExercise": reg_exercise_var.get()
        }

        gsr_features = {
            "SkinConductance": skin_conductance
        }

        payload = {
            "gsr_features": gsr_features,
            "clinical": {},
            "non_clinical": non_clinical
        }

        print("[Sending Non-Clinical Payload]", payload)

        response = requests.post("http://127.0.0.1:5000/predict_all", json=payload)
        if response.status_code == 200:
            result = response.json()
            prediction_label.config(
                text=f"Stress: {result['stress_level']} | PCOS Risk: {result['pcos_risk']} ({result['pcos_probability']}%)"
            )
        else:
            prediction_label.config(text="Error: Couldn't get prediction.")
    except Exception as e:
        print("[GUI ERROR]", e)
        import traceback
        traceback.print_exc()
        prediction_label.config(text=f"Error: {str(e)}")

# === GUI Layout ===
root = tk.Tk()
root.title("PCOS Detection System")
root.geometry("600x700")
root.configure(bg="#1e1e1e")

scrollable_frame = tk.Frame(root, bg="#1e1e1e")
scrollable_frame.grid(row=0, column=0, padx=10, pady=10)

tk.Label(scrollable_frame, text="Do you have clinical data?", fg="white", bg="#1e1e1e", font=("Arial", 14)).grid(row=0, column=0, pady=10)

ttkl = ttk.Button(scrollable_frame, text="Yes, I have clinical data", command=lambda: [non_clinical_frame.grid_remove(), clinical_frame.grid(row=4, column=0, padx=20, pady=10, sticky="w")])
ttkl.grid(row=1, column=0, pady=5)

ttkn = ttk.Button(scrollable_frame, text="No, I don't have clinical data", command=lambda: [clinical_frame.grid_remove(), non_clinical_frame.grid(row=4, column=0, padx=20, pady=10, sticky="w")])
ttkn.grid(row=2, column=0, pady=5)

prediction_label = tk.Label(scrollable_frame, text="Prediction: --", fg="white", bg="#1e1e1e", font=("Arial", 16))
prediction_label.grid(row=3, column=0, pady=20)

# === Clinical Form ===
clinical_frame = tk.Frame(scrollable_frame, bg="#1e1e1e")
labels = ["FSH", "LH", "AMH", "TSH", "PRL", "Vitamin D3"]
clinical_entries = []

for i, text in enumerate(labels):
    tk.Label(clinical_frame, text=text + ":", fg="white", bg="#1e1e1e").grid(row=i, column=0, pady=5, sticky="w")
    entry = tk.Entry(clinical_frame)
    entry.grid(row=i, column=1, pady=5, padx=10)
    clinical_entries.append(entry)

fsh_entry, lh_entry, amh_entry, tsh_entry, prl_entry, vit_d3_entry = clinical_entries

submit_btn_c = ttk.Button(clinical_frame, text="Submit Clinical Data", command=submit_clinical_data)
submit_btn_c.grid(row=len(labels), column=0, columnspan=2, pady=10)

# === Non-Clinical Form ===
non_clinical_frame = tk.Frame(scrollable_frame, bg="#1e1e1e")
labels_nc = ["AGE", "Weight (Kg)", "Height (Cm)", "BMI", "CYCLE LENGTH"]
nonclinical_entries = []

for i, text in enumerate(labels_nc):
    tk.Label(non_clinical_frame, text=text + ":", fg="white", bg="#1e1e1e").grid(row=i, column=0, pady=5, sticky="w")
    entry = tk.Entry(non_clinical_frame)
    entry.grid(row=i, column=1, pady=5, padx=10)
    nonclinical_entries.append(entry)

age_entry, weight_entry, height_entry, bmi_entry, cycle_length_entry = nonclinical_entries

# === Binary Options (Yes/No) ===
weight_gain_var = tk.IntVar()
hair_growth_var = tk.IntVar()
skin_darkening_var = tk.IntVar()
hair_loss_var = tk.IntVar()
fast_food_var = tk.IntVar()
reg_exercise_var = tk.IntVar()

def add_yes_no(label, var, row):
    tk.Label(non_clinical_frame, text=label + ":", fg="white", bg="#1e1e1e").grid(row=row, column=0, pady=5, sticky="w")
    yes_button = tk.Radiobutton(non_clinical_frame, text="Yes", variable=var, value=1, bg="#1e1e1e", fg="white", selectcolor="#1e1e1e", activebackground="#1e1e1e")
    no_button = tk.Radiobutton(non_clinical_frame, text="No", variable=var, value=0, bg="#1e1e1e", fg="white", selectcolor="#1e1e1e", activebackground="#1e1e1e")
    yes_button.grid(row=row, column=1, sticky="w")
    no_button.grid(row=row, column=2, sticky="w")

start_row = len(labels_nc)
add_yes_no("Weight Gain", weight_gain_var, start_row)
add_yes_no("Unwanted Hair Growth", hair_growth_var, start_row + 1)
add_yes_no("Skin Darkening", skin_darkening_var, start_row + 2)
add_yes_no("Hair Loss", hair_loss_var, start_row + 3)
add_yes_no("Fast Food", fast_food_var, start_row + 4)
add_yes_no("Regular Exercise", reg_exercise_var, start_row + 5)

submit_btn_nc = ttk.Button(non_clinical_frame, text="Submit Non-Clinical Data", command=submit_non_clinical_data)
submit_btn_nc.grid(row=start_row + 6, column=0, columnspan=3, pady=10)

# Run the GUI
root.mainloop()