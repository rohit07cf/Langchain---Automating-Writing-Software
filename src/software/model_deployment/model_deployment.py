import torch

class ModelDeployment:
    def save_model(self, model):
        torch.save(model.state_dict(), 'saved_model.pth')
        print('Model saved successfully.')

    def load_model(self):
        model = Model()
        model.load_state_dict(torch.load('saved_model.pth'))
        print('Model loaded successfully.')
        return model

    def make_predictions(self, data):
        model = self.load_model()
        predictions = model(data)
        print('Predictions:', predictions)

    def export_model(self, format):
        model = self.load_model()
        if format == 'onnx':
            torch.onnx.export(model, torch.zeros(1, 3, 224, 224), 'model.onnx')
            print('Model exported in ONNX format.')
        elif format == 'tensorflow':
            dummy_input = torch.zeros(1, 3, 224, 224)
            torch.onnx.export(model, dummy_input, 'model.onnx')
            import tf2onnx
            tf_model = tf2onnx.convert.from_onnx_file('model.onnx')
            tf_model.save_model('model.pb')
            print('Model exported in TensorFlow format.')
        else:
            print('Invalid export format.')

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=3)
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x