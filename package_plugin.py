"""Package the Photoshop UXP plugin into an installable .ccx file.
A .ccx is a zip archive with a .ccx extension that Adobe Creative Cloud recognizes.
Double-clicking the generated .ccx file installs the plugin into Adobe Photoshop.
"""
import zipfile
from pathlib import Path

def package_uxp_plugin():
    root = Path(__file__).resolve().parent
    plugin_dir = root / "plugin"
    dist_dir = root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    out_ccx = dist_dir / "PNG_Layer_Decomposer.ccx"

    if not plugin_dir.exists():
        raise FileNotFoundError(f"Plugin directory not found: {plugin_dir}")

    print(f"Packaging UXP plugin from: {plugin_dir}")
    print(f"Target installer package: {out_ccx}")

    with zipfile.ZipFile(out_ccx, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in plugin_dir.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(plugin_dir)
                zf.write(f, arcname=str(rel_path))
                print(f"  + {rel_path}")

    print(f"\nSUCCESS: Plugin packaged successfully!")
    print(f"File size: {out_ccx.stat().st_size / 1024:.1f} KB")
    print(f"Path: {out_ccx}")

if __name__ == "__main__":
    package_uxp_plugin()

