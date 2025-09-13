# launcher_cli.py
import sys, time
from face_registry import FaceEncoder, FaceRegistry

def main():
    enc = FaceEncoder()
    reg = FaceRegistry(enc, min_frames=10)  # allow fewer samples

    print("\n=== Eye-Gaze Drone: Operator Setup (CLI) ===")
    print("1) Add new user")
    print("2) Select operator and launch")
    choice = input("Choose [1/2]: ").strip()

    if choice == "1":
        name = input("Enter new user name: ").strip()
        ok = reg.add_user_from_camera(name, cam_index=0, seconds=12)  # give it more time
        print("Added." if ok else "Failed.")
        return

    if choice == "2":
        users = reg.list_users()
        if not users:
            print("No users yet. Run option 1 first.")
            return
        print("Known users:", ", ".join(users))
        sel = input("Type operator exactly: ").strip()
        if sel not in users:
            print("Unknown user.")
            return
        # Launch your demo with target
        import subprocess, sys as _sys
        cmd = [_sys.executable, "elg_demo.py", "--target", sel, "--match-thresh", "0.50", "--det-interval", "6", "--debug"]
        print("Launching:", " ".join(cmd))
        subprocess.Popen(cmd)
        print("Started. Press Ctrl+C to exit this launcher.")
        time.sleep(2)

if __name__ == "__main__":
    main()
