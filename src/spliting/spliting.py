from sklearn.model_selection import train_test_split

class DataSpliting:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def spliting(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42,stratify=self.y)
        return X_train, X_test, y_train, y_test


