# Name: Philasande Ngubo
# Date: 20 April 2025
# Custom Nueral Network


from torchvision import datasets

def main():
    Data_DIR = "."
    train_data = datasets.FashionMNIST(Data_DIR, train = True, download = False)
    test_data = datasets.FashionMNIST(Data_DIR, train = False, download = False)

    print(train_data)
    print(test_data)

if __name__ == "__main__":
    main()