from collections import Counter
import matplotlib.pyplot as plt


def verify_split(path_train, path_val, n):
    images_tuple_train = tuple()
    images_tuple_val = tuple()
    n_str = str(n)
    with open(path_train + "0" + n_str + ".txt", 'r') as file:
        file_content_train = file.read()

    with open(path_val + n_str + ".txt", 'r') as file:
        file_content_val = file.read()

    content_tuple_train = tuple(item.strip() for item in file_content_train.split('\n'))
    content_tuple_train = content_tuple_train[:-1]
    for item in content_tuple_train:
        item_tuple = tuple(item.split(' '))
        (img_path, label) = item_tuple
        item_tuple = (img_path, int(label))
        images_tuple_train = images_tuple_train + (item_tuple,)

    labels_train = [label for _, label in images_tuple_train]
    train_freq = Counter(labels_train)
    print(train_freq)

    content_tuple_val = tuple(item.strip() for item in file_content_val.split('\n'))
    content_tuple_val = content_tuple_val[:-1]
    for item in content_tuple_val:
        item_tuple = tuple(item.split(' '))
        (img_path, label) = item_tuple
        item_tuple = (img_path, int(label))
        images_tuple_val = images_tuple_val + (item_tuple,)

    labels_val = [label for _, label in images_tuple_val]
    val_freq = Counter(labels_val)
    print(val_freq)

    all_labels = list(set(labels_train) | set(labels_val))
    all_labels.sort()
    print(all_labels)

    bar_width = 0.35
    r1 = range(len(all_labels))
    r2 = [x + bar_width for x in r1]

    train_freqs = [train_freq[label] for label in all_labels]
    val_freqs = [val_freq[label] for label in all_labels]

    plt.bar(r1, train_freqs, color='blue', width=bar_width, edgecolor='grey', label='Train')
    plt.bar(r2, val_freqs, color='red', width=bar_width, edgecolor='grey', label='Validation')

    plt.xlabel('Label', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.xticks([r + bar_width / 2 for r in range(len(all_labels))], all_labels)

    plt.legend()
    plt.show()


verify_split("D:/Facultate/ANUL 4/Licenta/Licenta/Proiect/splits/train",
             "D:/Facultate/ANUL 4/Licenta/Licenta/Proiect/splits/validation", 0)
