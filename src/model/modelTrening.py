import time


class ModelTraining:

    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.trained_model = None
        self.training_time = None

    def train(self):
        start_time = time.perf_counter()
        self.model.fit(self.X_train,self.y_train)
        end_time = time.perf_counter()
        self.training_time = end_time - start_time
        self.trained_model = self.model
        print("Model Training Completed!")
        print(f"Training Time: {self.training_time:.2f} seconds")

        return self.trained_model
