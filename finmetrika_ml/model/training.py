import torch
from tqdm import tqdm
from finmetrika_ml.utils import check_device, moveTo
from transformers import AutoTokenizer, AutoModel
from datasets import DatasetDict



class TrainNN:
    #TODO Not finished and not tested
    """Train a neural network.
    
    Args:
        model: Instantiated model class or a defined model architecture.
        training_dataloader (torch.data.utils.DataLoader): Dataloader for training.
        loss_fn (str): Loss function
        optimizer (): optimizer
        num_epochs (int): Number of epochs to train.
        device (str): Device on which to train the model. Use utils.check_device().
    """
    def __init__(self, 
                 model, 
                 training_dataloader:torch.utils.data.DataLoader, 
                 loss_fn:str, 
                 optimizer,
                 num_epochs:int, 
                 device:str) -> None:
        self.model = model
        self.training_dataloader = training_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device


    def train(self):
        # Send the model to device
        self.model = self.model.to(self.device)
        
        for epoch in tqdm(range(self.num_epochs)):
            self.train_epoch
        
        
    def train_epoch(self):
        # Training mode
        self.model = self.model.train()
        
        # initialize training loss for the epoch
        training_loss = 0.0
        
        for inputs, labels in tqdm(self.training_dataloader):
            inputs = moveTo(inputs, self.device)
            labels = moveTo(labels, self.device)
            
            # Reset the gradients
            self.optimizer.zero_grad()
            
            # Model predictions
            outputs = self.model(inputs)
            
            # Compute the loss
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            training_loss += loss.item()
        print(training_loss)
            



class FineTuneFtsExtraction:
    #TODO Not finished and not tested
    """Fine tune a model using feature extraction. Training is done on the
    hidden states as features, without modifying the pretrained model.
    
    Args:
        model_name_hf (str): Model name as shown on HuggingFace
        dataset_hf (DatasetDict): Dataset dictionary with minimal splits:
                                  'train', 'validation', 'test'
        use_hf (bool): Use transformers library for training. 
    """
    def __init__(self, 
                 model_name_hf, 
                 dataset_hf:DatasetDict,
                 use_hf:bool=True,         
                 ) -> None:
        
        self.model_name_hf = model_name_hf
        self.dataset_hf = dataset_hf
        self.use_hf = use_hf
        
        self.device = check_device()
        self.model = (AutoModel.from_pretrained(self.model_name_hf)
                               .to(self.device)
                      )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
    
    def extract_hidden_states(self):
        """Extract hidden states from the model to use as features in the
        fine tuning model.
        """
        # Send the inputs to the GPU
        inputs = {
            k:v.to(self.device) for k,v in batch.items() if k in self.tokenizer.model_input_names
            }

        # Extract last hidden state
        


def model_size(model):
    """Count the number of parameters in the model
    
    Args:
        model: Instantiated model class.
    """    
    return sum(t.numel() for t in model.parameters())

