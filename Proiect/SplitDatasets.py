import random

# --- SPLIT TO DATASETS ---
images_with_labels_path = "D:/Facultate/ANUL 4/Licenta/Proiect/splits/train"

images_folder_path = "D:/Facultate/ANUL 4/Licenta/Proiect/images/"

def split_train_validation_file_function(path, n):
    images_set = set()
    n_str = str(n)
    with open(path + n_str + ".txt", 'r') as file:
        file_content = file.read()

    content_tuple = tuple(item.strip() for item in file_content.split('\n'))
    for item in content_tuple:
        item_tuple = tuple(item.split(' '))
        images_set.add(item_tuple)

    # print(images_set)
    images_len = images_set.__len__()
    split_index = int(images_len * 0.8)
    images_train_set = set(random.sample(list(images_set), split_index))
    images_validation_set = images_set - images_train_set

    # print(images_train_set)

    with open('D:/Facultate/ANUL 4/Licenta/Proiect/splits/train' + '0' + n_str + '.txt', 'w') as file:
        for item in images_train_set:
            # print(item)
            if item[0] != '':
                file.write(str(item[0]) + ' ' + str(item[1]))
                file.write('\n')

    with open('D:/Facultate/ANUL 4/Licenta/Proiect/splits/validation' + n_str + '.txt', 'w') as file:
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