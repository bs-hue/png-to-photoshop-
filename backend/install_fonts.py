import os
import shutil
import winreg
from pathlib import Path

fonts_src = Path(r'e:\Antigravity project\photoshop plugin\backend\fonts')
local_appdata = os.environ.get('LOCALAPPDATA', '')
user_fonts_dir = Path(local_appdata) / 'Microsoft' / 'Windows' / 'Fonts'
user_fonts_dir.mkdir(parents=True, exist_ok=True)

reg_path = r"Software\Microsoft\Windows NT\CurrentVersion\Fonts"
key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_SET_VALUE)

installed_count = 0
for ttf in fonts_src.glob("*.ttf"):
    dest = user_fonts_dir / ttf.name
    try:
        shutil.copy2(ttf, dest)
        font_reg_name = f"{ttf.stem} (TrueType)"
        winreg.SetValueEx(key, font_reg_name, 0, winreg.REG_SZ, str(dest))
        print(f"Installed & registered: {ttf.name}")
        installed_count += 1
    except Exception as e:
        print(f"Failed {ttf.name}: {e}")

winreg.CloseKey(key)

# Broadcast WM_FONTCHANGE so running apps (like Photoshop) pick up the fonts immediately
try:
    import ctypes
    HWND_BROADCAST = 0xFFFF
    WM_FONTCHANGE = 0x001D
    SMTO_ABORTIFHUNG = 0x0002
    result = ctypes.c_long()
    ctypes.windll.user32.SendMessageTimeoutW(
        HWND_BROADCAST, WM_FONTCHANGE, 0, 0, SMTO_ABORTIFHUNG, 1000, ctypes.byref(result)
    )
    print("WM_FONTCHANGE broadcast sent successfully!")
except Exception as e:
    print("Could not send broadcast:", e)

print(f"Total fonts registered: {installed_count}")
