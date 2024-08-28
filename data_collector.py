import os
import pickle
from matplotlib import pyplot as plt

class DataCollector:
    def __init__(self, INIT, root):
        self.lists_path = "/Users/osiprovin/Desktop/work/ML:DL/GANS/CDGAN/LOSS_ACCURACY"
        self.g_losses_root = os.path.join(root, 'G_losses.pkl')
        self.g_accuracy_root = os.path.join(root, 'G_accuracy.pkl')

        self.d_real_losses_root = os.path.join(root, 'D_real_losses.pkl')
        self.d_real_accuracy_root = os.path.join(root, 'D_real_accuracy.pkl')

        self.d_fake_losses_root = os.path.join(root, 'D_fake_losses.pkl')
        self.d_fake_accuracy_root = os.path.join(root, 'D_fake_accuracy.pkl')

        if INIT:
            self.G_losses = []
            self.G_accuracy = []

            self.D_real_losses = []
            self.D_real_accuracy = []

            self.D_fake_losses = []
            self.D_fake_accuracy = []
        else:
            self.load()

    def load(self):
        self.G_losses = pickle.load(open(self.g_losses_root, 'rb'))
        self.G_accuracy = pickle.load(open(self.g_accuracy_root, 'rb'))

        self.D_real_losses = pickle.load(open(self.d_real_losses_root, 'rb'))
        self.D_real_accuracy = pickle.load(open(self.d_real_accuracy_root, 'rb'))

        self.D_fake_losses = pickle.load(open(self.d_fake_losses_root, 'rb'))
        self.D_fake_accuracy = pickle.load(open(self.d_fake_accuracy_root, 'rb'))

    def get_data(self):
        return self.G_losses, self.G_accuracy, self.D_real_losses, self.D_real_accuracy, self.D_fake_losses, self.D_fake_accuracy
    
    def plot(self, list_of_lists, labels, title):
        plt.figure()
        plt.title(title)
        for list_n, label in zip(list_of_lists, labels):
            plt.plot([i for i in range(len(list_n))], list_n, label=label)
        plt.legend()
        plt.show()

    def save_list(self, list, name):
        file_path = os.path.join(self.lists_path, name)
        with open(f'{file_path}.pkl', 'wb') as file:
            pickle.dump(list, file)

    def save(self):
        self.save_list(self.G_losses, "G_losses")
        self.save_list(self.G_accuracy, "G_accuracy")

        self.save_list(self.D_real_losses, "D_real_losses")
        self.save_list(self.D_real_accuracy, "D_real_accuracy")

        self.save_list(self.D_fake_losses, "D_fake_losses")
        self.save_list(self.D_fake_accuracy, "D_fake_accuracy")