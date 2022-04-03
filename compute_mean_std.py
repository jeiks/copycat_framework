from sys import argv, stderr, exit
from copycat.data import Dataset
from copycat import calculate_mean_std

if __name__ == '__main__':
    if len(argv) != 2:
        print(f"Use: {argv[0]} image_list.txt [or image_list.txt.bz2]", file=stderr)
        exit(1)

    data = Dataset(data_filenames={'train':argv[1]})
    mean, std = calculate_mean_std(dataset=data, verbose=True)
    print(f"Mean: {mean}, STD: {std}")
