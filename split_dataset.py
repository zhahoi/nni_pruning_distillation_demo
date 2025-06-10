import os
import shutil
import random

def split_dataset(source_dir, dest_dir, split_ratio=0.7):
    classes = ['cat', 'dog']
    for class_name in classes:
        src_class_path = os.path.join(source_dir, class_name)
        all_files = os.listdir(src_class_path)
        all_files = [f for f in all_files if os.path.isfile(os.path.join(src_class_path, f))]
        random.shuffle(all_files)

        split_index = int(len(all_files) * split_ratio)
        train_files = all_files[:split_index]
        val_files = all_files[split_index:]

        for phase, file_list in zip(['train', 'val'], [train_files, val_files]):
            dst_class_path = os.path.join(dest_dir, phase, class_name)
            os.makedirs(dst_class_path, exist_ok=True)
            for filename in file_list:
                src_file = os.path.join(src_class_path, filename)
                dst_file = os.path.join(dst_class_path, filename)
                shutil.copy2(src_file, dst_file)

    print(f"Dataset split completed. Saved to: {dest_dir}")

if __name__ == "__main__":
    source_directory = 'animals'
    destination_directory = 'dataset'
    split_ratio = 0.7  # 70% train, 30% val

    split_dataset(source_directory, destination_directory, split_ratio)
