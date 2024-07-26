import numpy as np
import torchvision
print('------------------------------------------------------------')
print("I am inside models/tcl/data/create_noise.py")
print('------------------------------------------------------------')
################# Symmetric noise #########################
def random_in_noise(targets, noise_ratio=0.2):
    print('targets --> ', targets[0:10])
    targets = np.copy(np.array(targets))
    print('targets copy --> ', targets[0:10])
    num_classes = len(np.unique(targets))
    print('unique num of classes', num_classes)
    _num = int(len(targets) * noise_ratio)
    print('_num --> ', _num)
    clean_labels = np.copy(targets)
    print('clean labels --> ', clean_labels[0:10])

    # to be more equal, every category can be processed separately
    # np.random.seed(0)
    indices = np.random.permutation(len(targets))
    print('indices --> ', indices[0:10])

    for i, idx in enumerate(indices):
        print('i --> ', i)
        print('idx --> ', idx)
        print('noise ration * len(targets) --> ', noise_ratio * len(targets))
        if i < noise_ratio * len(targets):
            print('targets[idx] before update --> ', targets[idx])
            targets[idx] = np.random.randint(num_classes, dtype=np.int32)
            print('targets[idx] after update --> ', targets[idx])


    noisy_labels = np.asarray(targets, dtype=np.int32)
    print('noisy labels --> ', noisy_labels[0:10])
    print(f'num_classes: {num_classes}, rate: {noise_ratio}, actual_rate: {np.mean(noisy_labels != clean_labels)}')
    return noisy_labels


################# Real in-distribution noise #########################
def real_in_noise_cifar10(targets, noise_ratio=0.2):
    # to be more equal, every category can be processed separately
    # np.random.seed(0)

    targets = np.array(targets)
    num_classes = len(np.unique(targets))
    _num = int(len(targets) * noise_ratio)
    clean_labels = np.copy(targets)

    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise
    indices = np.random.permutation(len(targets))

    for i, idx in enumerate(indices):
        if i < int(noise_ratio * len(targets)):
            targets[idx] = transition[clean_labels[idx]]

    noisy_labels = np.asarray(targets, dtype=np.int32)
    print(f'num_classes: {num_classes}, rate: {noise_ratio}, actual_rate: {np.mean(noisy_labels != clean_labels)}')
    return noisy_labels


if __name__ == '__main__':

    dataset_name = 'cifar10'
    # dataset_name = 'cifar100'
    datasets = {'cifar10': torchvision.datasets.CIFAR10, 'cifar100': torchvision.datasets.CIFAR100}
    dataset = datasets[dataset_name]('/home/zzhuang/DATASET/clustering', train=True)
    targets = np.asarray(dataset.targets)
    # for noise_rate in [0.2, 0.5, 0.8, 0.9]:
    #     noisy_labels = random_in_noise(targets, noise_ratio=noise_rate)
    #     np.save(f'sym_noise_{dataset_name}_{int(noise_rate * 100)}', noisy_labels)
    for noise_rate in [0.4]:
        noisy_labels = real_in_noise_cifar10(targets, noise_ratio=noise_rate)
        np.save(f'asym_noise_{dataset_name}_{int(noise_rate * 100)}', noisy_labels)