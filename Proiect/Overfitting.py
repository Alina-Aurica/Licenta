import random

# --- SPLIT TO DATASETS ---
images_with_labels_path = "./splits/train"

images_folder_path = "./images/"

def split_train_validation_file_overfitting_function(path, n):
    images_set = set()
    n_str = str(n)
    with open(path + n_str + ".txt", 'r') as file:
        file_content = file.read()

    content_tuple = tuple(item.strip() for item in file_content.split('\n'))
    for item in content_tuple:
        item_tuple = tuple(item.split(' '))
        images_set.add(item_tuple)

    images_train_set = set(random.sample(list(images_set), 4))

    with open('./splits/overfit' + n_str + '.txt', 'w') as file:
        for item in images_train_set:
            # print(item)
            if item[0] != '':
                file.write(str(item[0]) + ' ' + str(item[1]))
                file.write('\n')


split_train_validation_file_overfitting_function(images_with_labels_path, 0)

