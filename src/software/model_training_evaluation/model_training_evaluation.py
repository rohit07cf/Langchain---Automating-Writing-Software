import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset_preprocessing.dataset_preprocessing import DatasetPreprocessing
from architecture_search.architecture_search import ArchitectureSearch
from hyperparameter_search.hyperparameter_search import HyperparameterSearch
from model_deployment.model_deployment import ModelDeployment

class ModelTrainingEvaluation:
    def __init__(self):
        self.selected_architecture = None
        self.selected_hyperparameters = None

    def train_model(self):
        # Load and preprocess the dataset
        dataset_preprocessing = DatasetPreprocessing()
        dataset = dataset_preprocessing.load_dataset()
        dataset = dataset_preprocessing.preprocess_dataset(dataset)
        train_dataset, val_dataset, test_dataset = dataset_preprocessing.split_dataset(dataset)

        # Search for the best architecture
        architecture_search = ArchitectureSearch()
        self.selected_architecture = architecture_search.search_architecture()

        # Search for the best hyperparameters
        hyperparameter_search = HyperparameterSearch()
        self.selected_hyperparameters = hyperparameter_search.search_hyperparameters()

        # Train the model
        model = self.selected_architecture.build_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.selected_hyperparameters.learning_rate)
        train_loader = DataLoader(train_dataset, batch_size=self.selected_hyperparameters.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.selected_hyperparameters.batch_size, shuffle=False)

        for epoch in range(self.selected_hyperparameters.num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Evaluate the model on the validation set
            val_loss = 0.0
            val_correct = 0
            total = 0
            model.eval()

            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_loss += criterion(outputs, labels).item()
                    val_correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            val_accuracy = val_correct / total
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}/{self.selected_hyperparameters.num_epochs}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")

        model.training = True

    def evaluate_model(self):
        # Load and preprocess the dataset
        dataset_preprocessing = DatasetPreprocessing()
        dataset = dataset_preprocessing.load_dataset()
        dataset = dataset_preprocessing.preprocess_dataset(dataset)
        _, _, test_dataset = dataset_preprocessing.split_dataset(dataset)

        # Load the trained model
        model = self.selected_architecture.build_model()
        model.load_state_dict(torch.load("model.pth"))
        model.eval()

        # Evaluate the model on the testing set
        test_loader = DataLoader(test_dataset, batch_size=self.selected_hyperparameters.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_loss += criterion(outputs, labels).item()
                test_correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_accuracy = test_correct / total
        test_loss /= len(test_loader)

        print("Evaluation Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

    def select_optimization_algorithm(self):
        optimization_algorithm = input("Select the optimization algorithm for training (Adam, SGD, etc.): ")
        # Set the optimization algorithm for training the model

    def specify_training_duration(self):
        training_duration = int(input("Specify the training duration (in epochs) or enter 0 for early stopping: "))
        # Set the training duration or early stopping criteria

    def display_real_time_metrics(self):
        real_time_metrics = RealTimeMetrics()
        real_time_metrics.display_metrics()
        # Display real-time training metrics and visualizations during the training process

    def save_model(self, model):
        torch.save(model.state_dict(), "model.pth")
        # Save the trained model for future use

    def load_model(self):
        model = self.selected_architecture.build_model()
        model.load_state_dict(torch.load("model.pth"))
        model.eval()
        return model
        # Load a saved model

    def make_predictions(self, data):
        model = self.load_model()
        # Make predictions on new data using the loaded model

    def export_model(self, format):
        model = self.load_model()
        # Export the model in formats compatible with different deployment environments

model_training_evaluation = ModelTrainingEvaluation()
model_training_evaluation.train_model()
model_training_evaluation.evaluate_model()