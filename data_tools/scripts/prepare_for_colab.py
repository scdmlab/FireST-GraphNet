"""
Google Drive 上传准备脚本

此脚本帮助准备数据集以上传到 Google Drive 用于 Colab 训练。
提供两种上传方式：
1. 压缩后上传（更快，但需要在 Colab 解压）
2. 直接上传原始文件（更慢，但 Colab 可直接使用）

使用方法：
    python prepare_for_colab.py --mode compress
    python prepare_for_colab.py --mode direct
"""

import argparse
import zipfile
from pathlib import Path
import shutil
import json


def get_dir_size(path):
    """计算目录大小（MB）"""
    total = 0
    for entry in Path(path).rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / (1024 * 1024)  # Convert to MB


def compress_dataset(data_dir, output_zip):
    """压缩数据集目录"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        return False
    
    print(f"\n[INFO] Compressing {data_dir}...")
    print(f"       Size: {get_dir_size(data_dir):.1f} MB")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in data_path.glob('*.npy'):
            zipf.write(file, arcname=file.name)
            print(f"  [OK] Added {file.name}")
    
    zip_size = Path(output_zip).stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Created {output_zip}")
    print(f"          Size: {zip_size:.1f} MB")
    return True


def create_file_manifest(data_dirs, output_json):
    """创建文件清单（用于验证上传完整性）"""
    manifest = {}
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            continue
        
        split_name = data_path.name  # dataset_train, dataset_val, etc.
        manifest[split_name] = {}
        
        for file in data_path.glob('*.npy'):
            file_size = file.stat().st_size / (1024 * 1024)  # MB
            manifest[split_name][file.name] = {
                'size_mb': round(file_size, 2),
                'path': str(file.relative_to(data_path.parent))
            }
    
    with open(output_json, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n[INFO] File manifest created: {output_json}")
    return manifest


def print_colab_instructions(mode, output_dir):
    """打印 Colab 使用说明"""
    print("\n" + "="*70)
    print("COLAB USAGE INSTRUCTIONS")
    print("="*70)
    
    if mode == 'compress':
        print("\n1. Upload compressed files to Google Drive:")
        print("   - Create folder: MyDrive/FireST-GraphNet/")
        print(f"   - Upload files from: {output_dir}")
        print("     * dataset_train.zip")
        print("     * dataset_val.zip")
        print("     * (optional) dataset_test.zip")
        
        print("\n2. In Colab notebook, mount Drive and extract:")
        print("""
```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
import os

# Extract train set
with zipfile.ZipFile('/content/drive/MyDrive/FireST-GraphNet/dataset_train.zip', 'r') as z:
    z.extractall('/content/dataset_train')

# Extract val set
with zipfile.ZipFile('/content/drive/MyDrive/FireST-GraphNet/dataset_val.zip', 'r') as z:
    z.extractall('/content/dataset_val')

print("Data extracted!")
print(f"Train files: {os.listdir('/content/dataset_train')}")
print(f"Val files: {os.listdir('/content/dataset_val')}")
```
""")
    
    else:  # direct
        print("\n1. Upload dataset folders to Google Drive:")
        print("   - Create folder: MyDrive/FireST-GraphNet/")
        print("   - Upload entire folders:")
        print("     * dataset_train/")
        print("     * dataset_val/")
        print("     * (optional) dataset_test/")
        
        print("\n2. In Colab notebook, mount Drive and use directly:")
        print("""
```python
from google.colab import drive
drive.mount('/content/drive')

# Point data paths to Drive
data_root = '/content/drive/MyDrive/FireST-GraphNet'

# Verify data
import os
train_files = os.listdir(f'{data_root}/dataset_train')
val_files = os.listdir(f'{data_root}/dataset_val')

print(f"Train files: {train_files}")
print(f"Val files: {val_files}")
```
""")
    
    print("\n3. Update config to use Colab paths:")
    print("""
```yaml
data:
  root: /content/dataset_train  # if extracted to /content
  # or
  root: /content/drive/MyDrive/FireST-GraphNet/dataset_train  # if direct from Drive
```
""")
    
    print("\n" + "="*70)
    print("TIP: Extracting to /content is faster than reading from Drive!")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Google Colab upload"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['compress', 'direct', 'both'],
        default='compress',
        help="Upload mode: compress (create zips), direct (upload folders), or both"
    )
    parser.add_argument(
        '--project-dir',
        type=str,
        default='.',
        help="FireST-GraphNet project directory"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./colab_upload',
        help="Output directory for compressed files"
    )
    
    args = parser.parse_args()
    
    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    print("="*70)
    print("GOOGLE COLAB DATA PREPARATION")
    print("="*70)
    print(f"\nProject directory: {project_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Mode:              {args.mode}\n")
    
    # 检查数据集目录
    data_dirs = []
    for split in ['train', 'val', 'test']:
        data_dir = project_dir / f'dataset_{split}'
        if data_dir.exists():
            data_dirs.append(data_dir)
            size_mb = get_dir_size(data_dir)
            print(f"[OK] Found dataset_{split}/  ({size_mb:.1f} MB)")
        else:
            if split != 'test':  # test 是可选的
                print(f"[WARN] dataset_{split}/ not found (${data_dir})")
    
    if not data_dirs:
        print("\n[ERROR] No dataset directories found!")
        print("Please ensure dataset_train/ and dataset_val/ exist.")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建文件清单
    manifest_file = output_dir / 'file_manifest.json'
    manifest = create_file_manifest(data_dirs, manifest_file)
    
    # 计算总大小
    total_size = sum(
        sum(info['size_mb'] for info in files.values())
        for files in manifest.values()
    )
    print(f"\n[INFO] Total dataset size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    
    # 根据模式处理
    if args.mode in ['compress', 'both']:
        print("\n" + "="*70)
        print("CREATING COMPRESSED ARCHIVES")
        print("="*70)
        
        for data_dir in data_dirs:
            split_name = data_dir.name  # dataset_train, dataset_val, etc.
            zip_name = f"{split_name}.zip"
            zip_path = output_dir / zip_name
            
            success = compress_dataset(data_dir, zip_path)
            if not success:
                print(f"[ERROR] Failed to compress {data_dir}")
                continue
        
        # 计算压缩后总大小
        compressed_size = sum(
            Path(output_dir / f"{d.name}.zip").stat().st_size / (1024 * 1024)
            for d in data_dirs
            if (output_dir / f"{d.name}.zip").exists()
        )
        
        print(f"\n[INFO] Total compressed size: {compressed_size:.1f} MB ({compressed_size/1024:.2f} GB)")
        print(f"[INFO] Compression ratio: {(compressed_size/total_size)*100:.1f}%")
        
        print_colab_instructions('compress', output_dir)
    
    if args.mode == 'direct':
        print("\n" + "="*70)
        print("DIRECT UPLOAD MODE")
        print("="*70)
        print("\n[INFO] In direct mode, upload the entire dataset folders to Google Drive.")
        print("       No compression needed - files will be used as-is in Colab.")
        
        print_colab_instructions('direct', project_dir)
    
    # 保存上传说明到文本文件
    instructions_file = output_dir / 'UPLOAD_INSTRUCTIONS.txt'
    with open(instructions_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GOOGLE COLAB UPLOAD INSTRUCTIONS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {Path.cwd()}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)\n\n")
        
        if args.mode in ['compress', 'both']:
            f.write("FILES TO UPLOAD:\n")
            for data_dir in data_dirs:
                zip_file = output_dir / f"{data_dir.name}.zip"
                if zip_file.exists():
                    size = zip_file.stat().st_size / (1024 * 1024)
                    f.write(f"  - {zip_file.name} ({size:.1f} MB)\n")
            f.write("\n")
        
        f.write("UPLOAD DESTINATION:\n")
        f.write("  Google Drive/My Drive/FireST-GraphNet/\n\n")
        
        f.write("See this file for detailed Colab code snippets.\n")
    
    print(f"\n[SUCCESS] Preparation complete!")
    print(f"[INFO] Upload instructions saved to: {instructions_file}")
    print(f"\n[NEXT] Upload the files/folders to Google Drive, then use them in Colab.")


if __name__ == "__main__":
    main()
