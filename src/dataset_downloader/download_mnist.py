#!/usr/bin/env python3
import os
import argparse
import requests
from tqdm import tqdm
from rich import print
import gzip
import numpy as np
from PIL import Image
import struct

def download_file(url: str, dest_path: str) -> None:
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print(f"[red]ERROR:[/red] Something went wrong downloading [yellow]{dest_path}[/yellow].")
    else:
        print(f"[green]Download complete:[/green] {dest_path}")

def decompress_file(file_path: str) -> str:
    if not file_path.endswith('.gz'):
        return file_path

    out_path = file_path.replace('.gz', '')
    if os.path.exists(out_path):
        print(f"[yellow]File already decompressed:[/yellow] {out_path}")
        return out_path

    with gzip.open(file_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            f_out.write(f_in.read())

    print(f"[green]Decompressed:[/green] {out_path}")
    return out_path

def read_idx_images(file_path: str) -> np.ndarray:
    with open(file_path, 'rb') as f:
        data = f.read()
    # first 16 bytes are header information
    magic, num_images, rows, cols = struct.unpack(">IIII", data[:16])
    images = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(num_images, rows, cols)
    return images

def read_idx_labels(file_path: str) -> np.ndarray:
    with open(file_path, 'rb') as f:
        data = f.read()
    # first 8 bytes are header information
    magic, num_labels = struct.unpack(">II", data[:8])
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels

def save_images(images: np.ndarray, labels: np.ndarray, output_dir: str, dataset_type: str):
    """
    Save images as PNG files in subdirectories according to their label
    """
    base_dir = os.path.join(output_dir, "mnist", dataset_type)
    for label in range(10):
        os.makedirs(os.path.join(base_dir, str(label)), exist_ok=True)

    for idx, (img, label) in enumerate(tqdm(zip(images, labels), total=len(images), desc=f"Saving {dataset_type} images")):
        img_pil = Image.fromarray(img, mode='L')
        img_filename = os.path.join(base_dir, str(label), f"image_{idx:05d}.png")
        img_pil.save(img_filename)

def main():
    parser = argparse.ArgumentParser(description="Download, unzip, and convert the MNIST dataset to images.")
    parser.add_argument("dest_dir", help="Destination directory for the MNIST dataset.")
    args = parser.parse_args()

    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
        print(f"[green]Created directory:[/green] {args.dest_dir}")
    else:
        print(f"[green]Using directory:[/green] {args.dest_dir}")

    # MNIST files and their URLs
    files = {
        "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
    }

    # download each file if it doesn't already exist
    for filename, url in files.items():
        dest_path = os.path.join(args.dest_dir, filename)
        if os.path.exists(dest_path):
            print(f"[yellow]Skipping download:[/yellow] {dest_path} already exists.")
        else:
            print(f"[blue]Downloading:[/blue] {filename}")
            download_file(url, dest_path)

    # decompress the downloaded files
    decompressed_files = {}
    for filename in files.keys():
        gz_path = os.path.join(args.dest_dir, filename)
        decompressed_path = decompress_file(gz_path)
        decompressed_files[filename[:-3]] = decompressed_path

    # set file paths for training and test data
    train_images_file = decompressed_files.get("train-images-idx3-ubyte")
    train_labels_file = decompressed_files.get("train-labels-idx1-ubyte")
    test_images_file = decompressed_files.get("t10k-images-idx3-ubyte")
    test_labels_file = decompressed_files.get("t10k-labels-idx1-ubyte")

    print("[blue]Processing training data...[/blue]")
    train_images = read_idx_images(train_images_file)
    train_labels = read_idx_labels(train_labels_file)
    save_images(train_images, train_labels, args.dest_dir, dataset_type="train")

    print("[blue]Processing test data...[/blue]")
    test_images = read_idx_images(test_images_file)
    test_labels = read_idx_labels(test_labels_file)
    save_images(test_images, test_labels, args.dest_dir, dataset_type="test")

    print("[green]All tasks completed successfully.[/green]")

if __name__ == "__main__":
    main()
