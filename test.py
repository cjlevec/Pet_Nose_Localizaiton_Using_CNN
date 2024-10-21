import torch
import numpy as np
import argparse
from model import SnoutNet
from dataset import SnoutNoseDataset, DataLoader, transform1

# Added so that I can run the code on PyCharm and Colab
ColabPath = "/content/drive/My Drive/ELEC 475 Lab 2 CO/oxford-iiit-pet-noses/"
HomePath = "/Users/christianlevec/Documents/475 Lab 2/oxford-iiit-pet-noses/"
ScottPath = "/Users/scottdoggett/PycharmProjects/Pet_Nose_Localizaiton_Using_CNN/oxford-iiit-pet-noses/"

def main():


    # Read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='weights', type=str, help='parameter file (.pth)')
    args = argParser.parse_args()

    weights_file = None
    if args.s != None:
        weights_file = args.s

    print('\t\tweights file = ', weights_file)


    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    # Create test set
    testSet = SnoutNoseDataset(HomePath+"test_noses.txt", HomePath+"images-original/images", transform=transform1)
    test_dataloader = DataLoader(testSet, batch_size=61, shuffle=False)

    # Evaluate the model
    model = SnoutNet()
    model.load_state_dict(torch.load("weights.pth", map_location=torch.device('cpu')))
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