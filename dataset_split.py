import os
import shutil
import click
from sklearn.model_selection import train_test_split

def load_data(image_folder, annotation_folder):
    """
    Load and match images with their corresponding annotations based on filename(without extension)
    """

    #use the filename as keys of the dictionary
    image_files = {os.path.splitext(f)[0]: os.path.join(image_folder, f) 
                    for f in os.listdir(image_folder)}
    annotation_files = {os.path.splitext(f)[0]: os.path.join(annotation_folder, f) 
                        for f in os.listdir(annotation_folder)}

    common_keys = image_files.keys() & annotation_files.keys() #AND operation to check the matching of images and annotations
    missing_images = annotation_files.keys() - common_keys
    missing_annotations = image_files.keys() - common_keys

    if missing_images:
        print("\n Missing Images (annotations exist but no image found) for:")
        for key in sorted(missing_images):
            print(f" - {annotation_files[key]}")

    if missing_annotations:
        print("\n Missing Annotations (Images exist but no annotation found) for:")
        for key in sorted(missing_annotations):
            print(f" - {image_files[key]}")

    paired_data = [(image_files[k], annotation_files[k]) for k in sorted(common_keys)]
    
    if not paired_data:
        raise ValueError("ERROR: No valid image-annotation pairs found! Please check your dataset.")

    return paired_data

def split_dataset(dataset, train_ratio, val_ratio, test_ratio):

    train_set, val_test_set = train_test_split(dataset, train_size=train_ratio, random_state=42, shuffle=True, stratify=None) #split train and val+test
    #val and test split
    val_set, test_set = train_test_split(val_test_set, train_size=val_ratio / (val_ratio + test_ratio), test_size=test_ratio / (val_ratio + test_ratio), random_state=42, shuffle=True, stratify=None)
    
    print(f"INFO: Train set size: {len(train_set)}, Validation set size: {len(val_set)}, Test set size: {len(test_set)}")

    return train_set, val_set, test_set

def save_dataset(dataset, root_dir, dataset_type):
    image_folder = os.path.join(root_dir, dataset_type, 'images')
    annotation_folder = os.path.join(root_dir, dataset_type, 'annotations')
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(annotation_folder, exist_ok=True)

    for image_path, annotation_path in dataset:
        shutil.copy(image_path, image_folder)
        shutil.copy(annotation_path, annotation_folder)

@click.command()
@click.argument('images', type=click.Path(exists=True))
@click.argument('annotations', type=click.Path(exists=True))
@click.option('--dataset_split', nargs=3, type=float, default=[0.7, 0.2, 0.1], help='Train, validation and test ratios')
@click.option('--output_folder', default='dataset_after_split', help='Output folder')
def main(images, annotations, output_folder, dataset_split):
    
    train_ratio, val_ratio, test_ratio = dataset_split
    assert abs(sum(dataset_split) - 1) < 1e-5, " Train, validation and test ratios must sum to 1"
        

    dataset = load_data(images, annotations) # list of tuples (image_path, annotation_path)

    train_set, val_set, test_set = split_dataset(dataset, train_ratio, val_ratio, test_ratio)

    save_dataset(train_set, output_folder, 'train')
    save_dataset(val_set, output_folder, 'val')
    save_dataset(test_set, output_folder, 'test')

    print(f"INFO: Dataset saved in {output_folder}")

if __name__ == '__main__':
    main()


'''
usage example: python dataset_split.py <image_path> <annotation_path> --dataset_split 0.7 0.2 0.1 --output_folder <output_dir_folder or path>
'''
