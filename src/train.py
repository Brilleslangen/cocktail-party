

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.validate(val_loader)

    def validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                # Add validation metrics here
