"""
个人模型导出器 (供 Streamlit 调用)
==================================
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent


def export_mobile_package(model_path: Path) -> str:
    """
    导出 Mobile 格式并打包成可分发的 zip.

    Returns:
        zip 包路径
    """
    output_dir = ROOT / "exported"
    output_dir.mkdir(exist_ok=True)

    # 调用主导出脚本
    export_script = ROOT / "export" / "export_v16_6_50m.py"
    subprocess.run([
        sys.executable, str(export_script),
        "--ckpt", str(model_path),
        "--format", "mobile",
        "--output", str(output_dir),
    ], check=True)

    # 找到生成的 mobile 文件
    ptl_file = output_dir / "v16_6_50m_mobile.ptl"
    if not ptl_file.exists():
        ptl_file = output_dir / "hibs_0_16_mobile.ptl"

    if not ptl_file.exists():
        raise FileNotFoundError("Mobile 导出失败, 找不到 .ptl 文件")

    # 重命名为更友好的名字
    final_ptl = output_dir / "hibs_model.ptl"
    shutil.copy(ptl_file, final_ptl)

    # 打包 zip
    zip_path = output_dir / "HibsApp_Model.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(final_ptl, "hibs_model.ptl")
        # 附 README
        readme_content = """# Hibs Mobile Model

把这个文件复制到手机:

**iOS**: 使用 iTunes/Files 拖入 Hibs App 的 Documents 目录
**Android**: 放到 /sdcard/Android/data/com.hibs.app/files/

模型完全本地运行, 无需联网.
"""
        zf.writestr("README.txt", readme_content)

    return str(zip_path)
