import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def get_dataset_dfs(dataset_path: str) -> pd.DataFrame:
    # the dataset path is a directory containing the dataset files
    # the structure must be:
    # dataset_path
    # |-- dataset_name
    # |   |-- train
    # |   |   |-- class_1
    # |   |       |-- img_1.jpg
    # |   |       |-- img_2.jpg
    # |   |       |-- ...
    # |   |   |-- class_2
    # |   |   |-- ...
    # |   |-- val (optional)
    # |   |   |-- class_1
    # |   |       |-- img_1.jpg
    # |   |       |-- img_2.jpg
    # |   |       |-- ...
    # |   |   |-- class_2
    # |   |   |-- ...
    # |   |-- test (optional)
    # |   |   |-- class_1
    # |   |       |-- img_1.jpg
    # |   |       |-- img_2.jpg
    # |   |       |-- ...
    # |   |   |-- class_2
    # |   |   |-- ...

    # Returns a dict of dataframes, one for each split (train, val, test)
    # Each dataframe contains the following columns:
    # - image_path: the absolute path to the image
    # - class_name: the name of the class (the folder name)

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise ValueError(f"Dataset path {dataset_path} does not exist.")
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset path {dataset_path} is not a directory.")

    # find the dataset name (first and only folder in the dataset path)
    dataset_name = next(dataset_path.iterdir()).name
    dataset_path = dataset_path / dataset_name

    dfs = {}
    for split in tqdm(['train', 'val', 'test'], desc='Loading dataset splits', unit='split'):
        split_path = dataset_path / split
        if not split_path.exists():
            continue

        class_names = [d.name for d in split_path.iterdir() if d.is_dir()]

        data = []
        for class_name in class_names:
            class_path = split_path / class_name
            image_paths = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))  # assuming all images are jpg or png
            data.extend([(str(image_path), class_name) for image_path in image_paths])

        # create a dataframe from the list of tuples
        df = pd.DataFrame(data, columns=['image_path', 'class_name']).reset_index(drop=True)
        df['image_path'] = df['image_path'].apply(lambda x: os.path.abspath(x))  # convert to absolute path
        df['class_name'] = df['class_name'].astype('category')
        df['class_id'] = df['class_name'].cat.codes
        df['class_id'] = df['class_id'].astype('int')
        dfs[split] = df

    return dfs

def load_images(image_paths: list, max_workers: int = 8, mode: str = 'RGB', target_size=(28, 28)) -> list:

    from concurrent.futures import ThreadPoolExecutor
    import cv2

    def load_image(image_path):
        # load image using cv2
        img = cv2.imread(image_path, cv2.IMREAD_COLOR if mode == 'RGB' else cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image {image_path} could not be loaded.")
        img = cv2.resize(img, target_size)
        if mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    images = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(image_paths), desc='Loading images', unit='img') as pbar:
            future_to_path = {executor.submit(load_image, path): path for path in image_paths}

            for future in future_to_path:
                try:
                    img = future.result()
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {future_to_path[future]}: {e}")
                pbar.update(1)

    return images