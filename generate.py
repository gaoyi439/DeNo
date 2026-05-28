import numpy as np
import sklearn
import torch

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

def generate_pml_label(basic_labels, noisy_level):
    """
    Generate Partial Multi-Label (PML) data by adding noisy labels to the original labels.

    :param basic_labels: A binary matrix (n, k), where n is the number of samples 
                         and k is the number of classes. Each sample may have multiple true labels.
    :param noisy_level: A float (0~1) representing the probability of introducing additional noisy labels.
    :return: A modified label matrix (n, k) with added noisy labels.
    """
    n, k = basic_labels.shape
    pml_labels = np.copy(basic_labels)  # Copy the original labels to avoid modifying the input data

    for idx in range(n):
        # Calculate the number of noisy labels to add
        num_noisy_labels = int(noisy_level * (k - np.sum(basic_labels[idx])))

        # Find the candidate labels that are currently unassigned (0 values)
        available_labels = np.where(basic_labels[idx] == 0)[0]

        # Randomly select noisy labels and set them to 1
        noisy_labels = np.random.choice(available_labels, min(num_noisy_labels, len(available_labels)), replace=False)
        pml_labels[idx, noisy_labels] = 1

    return pml_labels



def generate_compl_labels(pml_labels, num_com):
    """
    Generate complementary labels based on Partial Multi-Label (PML) data.

    :param pml_labels: A binary matrix (n, k), where n is the number of samples 
                       and k is the number of classes. Each sample may have multiple labels (1s).
    :param num_com: The number of complementary labels to assign per sample.
    :return: A binary matrix (n, k) representing complementary labels for each instance.
    """
    n, k = pml_labels.shape
    comp_Y = np.zeros([n, k])  # Initialize complementary label matrix with all zeros
    labels_hat = np.array(1 - pml_labels, dtype=bool)  # Only consider labels not in PML (i.e., 0s in pml_labels)

    for idx in range(n):
        # Get candidate labels that are not part of the PML labels
        candidates = np.arange(k)  # All possible class indices
        available_labels = candidates[labels_hat[idx]]  # Filter out labels present in PML

        # Ensure that num_com does not exceed available complementary labels
        num_com_actual = min(num_com, len(available_labels))
        if num_com_actual > 0:
            selected_complementary = np.random.choice(available_labels, num_com_actual, replace=False)
            comp_Y[idx, selected_complementary] = 1  # Mark the selected labels as complementary

    return comp_Y


# generate complementary labels from existing pml labels
datasets = ["YeastCC", "YeastBP", "music_emotion", "music_style"]
com_file = ["data/YeastCC_com_label.csv","data/YeastBP_com_label.csv","data/music_emotion_com_label.csv","data/music_style_com_label.csv"]
pml_file = ["data/YeastCC_pml_label.csv","data/YeastBP_pml_label.csv","data/music_emotion_pml_label.csv","data/music_style_pml_label.csv"]

for i, dataset_name in enumerate(datasets):
    print(f"Processing {dataset_name}...")
    # Read existing pml labels
    pml_labels = np.genfromtxt(pml_file[i], delimiter=',')
    print(f"  PML labels shape: {pml_labels.shape}")
    
    # Generate complementary labels from pml labels
    com_labels = generate_compl_labels(pml_labels, 1)
    np.savetxt(com_file[i], com_labels, delimiter=',')
    print(f"  COM labels saved to {com_file[i]}")
    print()
