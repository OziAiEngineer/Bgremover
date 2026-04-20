"""
Cleanup script to remove unnecessary files for basic eraser functionality
Keeps only what's needed for LaMa eraser to work
"""
import os
import shutil
from pathlib import Path

def remove_path(path):
    """Remove file or directory"""
    if not path.exists():
        print(f"  ⏭️  Skip (not found): {path}")
        return
    
    try:
        if path.is_file():
            path.unlink()
            print(f"  ✅ Removed file: {path}")
        elif path.is_dir():
            shutil.rmtree(path)
            print(f"  ✅ Removed folder: {path}")
    except Exception as e:
        print(f"  ❌ Error removing {path}: {e}")

print("\n" + "="*70)
print("🧹 Cleaning up unnecessary files for eraser functionality")
print("="*70 + "\n")

# Root level removals
root_removals = [
    ".github",
    ".vscode",
    "assets",
    "docker",
    "scripts",
    "web_app",
    "output_folder",
    "build_docker.sh",
    "publish.sh",
    "main.py",
    "setup.py",
    "requirements-dev.txt",
]

print("📁 Removing root level files/folders...")
for item in root_removals:
    remove_path(Path(item))

# IOPaint removals
iopaint_removals = [
    "iopaint/api.py",
    "iopaint/batch_processing.py",
    "iopaint/benchmark.py",
    "iopaint/cli.py",
    "iopaint/download.py",
    "iopaint/installer.py",
    "iopaint/web_config.py",
    "iopaint/__main__.py",
    "iopaint/file_manager",
    "iopaint/plugins",
    "iopaint/tests",
]

print("\n📁 Removing unnecessary iopaint files/folders...")
for item in iopaint_removals:
    remove_path(Path(item))

# Model removals (keep only lama, base, utils)
model_removals = [
    "iopaint/model/anytext",
    "iopaint/model/brushnet",
    "iopaint/model/helper",
    "iopaint/model/original_sd_configs",
    "iopaint/model/power_paint",
    "iopaint/model/controlnet.py",
    "iopaint/model/ddim_sampler.py",
    "iopaint/model/fcf.py",
    "iopaint/model/instruct_pix2pix.py",
    "iopaint/model/kandinsky.py",
    "iopaint/model/ldm.py",
    "iopaint/model/manga.py",
    "iopaint/model/mat.py",
    "iopaint/model/mi_gan.py",
    "iopaint/model/opencv2.py",
    "iopaint/model/paint_by_example.py",
    "iopaint/model/plms_sampler.py",
    "iopaint/model/sd.py",
    "iopaint/model/sdxl.py",
    "iopaint/model/zits.py",
]

print("\n📁 Removing unnecessary model files/folders...")
for item in model_removals:
    remove_path(Path(item))

print("\n" + "="*70)
print("✨ Cleanup complete!")
print("="*70)
print("\n📦 Remaining essential files:")
print("  - simple_server.py (your eraser server)")
print("  - simple_eraser.html (your UI)")
print("  - iopaint/model/lama.py (LaMa eraser model)")
print("  - iopaint/model/base.py (base model)")
print("  - iopaint/model/utils.py (utilities)")
print("  - iopaint/helper.py (image helpers)")
print("  - iopaint/schema.py (data schemas)")
print("  - iopaint/model_manager.py (model manager)")
print("  - requirements.txt (dependencies)")
print("\n🚀 You can now run: python simple_server.py")
print("="*70 + "\n")
