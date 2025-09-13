import tkinter as tk
import cv2
import face_recognition
import os, pickle

DB_PATH = "face_db.pkl"

def load_db():
    return pickle.load(open(DB_PATH, "rb")) if os.path.exists(DB_PATH) else {}

def save_db(db):
    pickle.dump(db, open(DB_PATH, "wb"))

def add_user(name):
    cap = cv2.VideoCapture(0)
    embeddings = []
    print(f"[INFO] Recording face data for {name}. Look into the camera...")
    for _ in range(30):  # capture 30 frames
        ret, frame = cap.read()
        if not ret: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if encs:
            embeddings.append(encs[0])
    cap.release()
    if embeddings:
        db = load_db()
        db[name] = sum(embeddings) / len(embeddings)  # average embedding
        save_db(db)
        print(f"[INFO] User {name} added with {len(embeddings)} samples.")

def select_user():
    db = load_db()
    if not db: return None
    root = tk.Tk()
    root.title("Select Operator")
    selected = tk.StringVar()

    for user in db.keys():
        tk.Radiobutton(root, text=user, variable=selected, value=user).pack(anchor="w")

    def confirm():
        root.destroy()
    tk.Button(root, text="Confirm", command=confirm).pack()
    root.mainloop()
    return selected.get()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Drone Operator Interface")

    tk.Button(root, text="Add New User", command=lambda: add_user("NewUser")).pack(pady=10)
    tk.Button(root, text="Select Operator", command=select_user).pack(pady=10)

    root.mainloop()
