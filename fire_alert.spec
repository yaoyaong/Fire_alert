# PyInstaller spec for Fire Alert (CCTV) - run from repo root: pyinstaller fire_alert_exe\fire_alert.spec
# Output: fire_alert_exe\dist\Fire_Alert\Fire_Alert.exe (and writable streams/ uploads/ next to exe at runtime)

import os
import sys

# Repo root = parent of fire_alert_exe
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
REPO_ROOT = os.path.dirname(SPEC_DIR)
APP_DIR = os.path.join(REPO_ROOT, 'fire_alert_python')

# All read-only app data (static, HTML, JS, CSS, best.pt) go into bundle so they're in _MEIPASS when frozen
static_src = os.path.join(APP_DIR, 'static')
if not os.path.isdir(static_src):
    raise SystemExit(f'Static folder not found: {static_src}')

a = Analysis(
    [os.path.join(APP_DIR, 'app.py')],
    pathex=[APP_DIR],
    binaries=[],
    datas=[
        (static_src, 'static'),
    ],
    hiddenimports=[
        'flask', 'flask_cors', 'werkzeug.utils', 'pymysql', 'pymysql.cursors', 'bcrypt',
        'PIL', 'PIL.Image', 'cv2', 'numpy',
        'ultralytics', 'ultralytics.models', 'ultralytics.models.yolo',
        'torch', 'torchvision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Fire_Alert',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Fire_Alert',
)

# One-folder output: fire_alert_exe\dist\Fire_Alert\Fire_Alert.exe + _internal\
# streams/ and uploads/alerts/ are created next to the exe at runtime.
# Build: from repo root run  pyinstaller fire_alert_exe\fire_alert.spec
