import torch
import numpy as np
import argparse
from model import SnoutNet
from dataset import SnoutNoseDataset, DataLoader, transform1, transformNoise, transformFlip

# Added so that I can run the code on PyCharm and Colab
ColabPath = "/content/drive/My Drive/ELEC 475 Lab 2 CO/oxford-iiit-pet-noses/"
HomePath = "/Users/christianlevec/Documents/475 Lab 2/oxford-iiit-pet-noses/"
ScottPath = "/Users/scottdoggett/PycharmProjects/oxford-iiit-pet-noses/"

def main():


    # Read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-w', metavar='weights', type=str, help='parameter file (.pth)')
    argParser.add_argument('-a', metavar='augmentation', type=int, help='augmentation: 0=no augmentation, 1=flip, 2=noise')
    args = argParser.parse_args()

    weights_file = None
    if args.w != None:
        weights_file = args.w
    if args.a != None:
        augmentation = args.a
    else:
        augmentation = 0

    finalTransformation = transform1
    flipped = 0

    if augmentation == 1:
        flipped = 1
        finalTransformation = transformFlip
        print('\t\taugmentation: horizontal flip')
    elif augmentation == 2:
        finalTransformation = transformNoise
        print('\t\taugmentation: original')
    else:
        print('\t\taugmentation: none')

    print('\t\tweights file = ', weights_file)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    # Create test set
    #testSet = SnoutNoseDataset(HomePath+"test_noses.txt", HomePath+"images-original/images", transform=transform)
    testSet = SnoutNoseDataset(ScottPath+"test_noses.txt",
                               ScottPath+"images-original/images",
                               transform=finalTransformation,
                               flipped=flipped)

    test_dataloader = DataLoader(testSet, batch_size=61, shuffle=False)

    # Evaluate the model
    model = SnoutNet()
    model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Calculate Euclidean distance
            distances = []
            for predicted, target in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                # Euclidian distance calculation
                distance = np.linalg.norm(predicted - target)
                distances.append(distance)

    # Calculate statistics
    min_distance = np.min(distances)
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    std_distance = np.std(distances)

    print("Localization accuracy statistics:")
    print("Minimum distance:", min_distance)
    print("Mean distance:", mean_distance)
    print("Maximum distance:", max_distance)
    print("Standard deviation:", std_distance)

if __name__ == '__main__':
    main()