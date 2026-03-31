import tensorflow as tf
import numpy as np

class LiteModel:
    def __init__(self, model_path):
        print(f"⚡ Loading TFLite Model: {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def predict(self, input_data):
        # Ensure input is float32
        input_data = input_data.astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        return self.interpreter.get_tensor(self.output_details[0]['index'])