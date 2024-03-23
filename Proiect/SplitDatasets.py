import random
from sklearn.model_selection import train_test_split

# --- SPLIT TO DATASETS ---
images_with_labels_path = "D:/Facultate/ANUL 4/Licenta/Licenta/Proiect/splits/train"

images_folder_path = "D:/Facultate/ANUL 4/Licenta/Licenta/Proiect/images/"

def split_each_class(images_set):
    # print(images_set)
    dataset_list = list(images_set)
    paths = []
    labels = []
    for item in dataset_list:
        if len(item) == 2:
            path, label = item
            paths.append(path)
            labels.append(label)
        else:
            print(f"Skipping invalid item with {len(item)} elements: {item}")

    # Perform stratified split
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_set = list(zip(paths_train, labels_train))
    val_set = list(zip(paths_val, labels_val))

    return train_set, val_set


def split_train_validation_file_function(path, n):
    images_set = set()
    n_str = str(n)
    with open(path + n_str + ".txt", 'r') as file:
        file_content = file.read()

    content_tuple = tuple(item.strip() for item in file_content.split('\n'))
    for item in content_tuple:
        item_tuple = tuple(item.split(' '))
        images_set.add(item_tuple)

    # # print(images_set)
    # images_len = images_set.__len__()
    # split_index = int(images_len * 0.8)
    # images_train_set = set(random.sample(list(images_set), split_index))
    # images_validation_set = images_set - images_train_set

    images_train_set, images_validation_set = split_each_class(images_set)

    # print(images_train_set)

    with open('D:/Facultate/ANUL 4/Licenta/Licenta/Proiect/splits/train' + '0' + n_str + '.txt', 'w') as file:
        for item in images_train_set:
            # print(item)
            if item[0] != '':
                file.write(str(item[0]) + ' ' + str(item[1]))
                file.write('\n')

    with open('D:/Facultate/ANUL 4/Licenta/Licenta/Proiect/splits/validation' + n_str + '.txt', 'w') as file:
        for item in images_validation_set:
            if item[0] != '':
                file.write(str(item[0]) + ' ' + str(item[1]))
                file.write('\n')


split_train_validation_file_function(images_with_labels_path, 0)
split_train_validation_file_function(images_with_labels_path, 1)
split_train_validation_file_function(images_with_labels_path, 2)
split_train_validation_file_function(images_with_labels_path, 3)
split_train_validation_file_function(images_with_labels_path, 4)

# --- RETURN TUPLE OF DATASET ---

def return_tuple_of_dataset(path, n):
    images_tuple = tuple()
    n_str = str(n)
    with open(path + n_str + ".txt", 'r') as file:
        file_content = file.read()

    content_tuple = tuple(item.strip() for item in file_content.split('\n'))
    for item in content_tuple:
        item_tuple = tuple(item.split(' '))
        images_tuple = images_tuple + (item_tuple,)

    images_tuple = images_tuple[:-1]
    return images_tuple