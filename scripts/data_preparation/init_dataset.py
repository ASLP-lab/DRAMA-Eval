#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

def sync_directory(src_dir, dest_dir):
    if not src_dir.exists():
        print(f"⚠️ 警告: 源目录不存在 -> {src_dir}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 正在同步: {src_dir.name} -> {dest_dir}")
    for src_file in src_dir.rglob('*'):
        if src_file.is_file():
            rel_path = src_file.relative_to(src_dir)
            dest_file = dest_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)

def main():
    parser = argparse.ArgumentParser(description="Initialize DRAMA-Eval local dataset using HuggingFace downloads.")
    parser.add_argument("--hf_dir", required=True, type=str, help="Path to the downloaded HuggingFace Dataset folder")
    args = parser.parse_args()

    hf_path = Path(args.hf_dir).resolve()
    local_data_path = Path(__file__).resolve().parent.parent.parent / "data"

    print("==================================================")
    print("🚀 开始初始化 DRAMA-Eval 数据集...")
    print("==================================================")

    sync_directory(hf_path / "en" / "audios", local_data_path / "en" / "audios")
    sync_directory(hf_path / "zh" / "audios", local_data_path / "zh" / "audios")
    sync_directory(hf_path / "prompt_wav", local_data_path / "prompt_wav")

    print("\n🎉 数据集初始化完成！现在的 data/ 目录已经具备所有必要的音频文件，可以开始运行评测了。")

if __name__ == "__main__":
    main()
