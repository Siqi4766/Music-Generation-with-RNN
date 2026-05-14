# !pip install comet_ml > /dev/null 2>&1
import comet_ml
COMET_API_KEY = ""

import torch
import torch.nn as nn
import torch.optim as optim

# !pip install mitdeeplearning --quiet
import mitdeeplearning as mdl
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write
# !apt-get install abcmidi timidity > /dev/null 2>&1

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
assert COMET_API_KEY != "", "Please insert your Comet API Key"

# Download the dataset
songs = mdl.lab1.load_training_data()

# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs)

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")

# Create a mapping from character to unique index.
char2idx = {u: i for i, u in enumerate(vocab)}

# Create a mapping from indices to characters. 
idx2char = np.array(vocab)

def vectorize_string(string):
  """
  Convert a string of music notation characters into vectorized (i.e., numeric) representation.

  Each character is mapped through the global ``char2idx`` vocabulary lookup.
  The returned NumPy array can be used as model input or sampled into training
  batches.

  Args:
    string: String containing ABC music notation characters.

  Returns:
    A one-dimensional NumPy array.
  """
  return np.array([char2idx[char] for char in string])

vectorized_songs = vectorize_string(songs_joined)

def get_batch(vectorized_songs, seq_length, batch_size):
    """
    Sample a random batch of input and target sequences for next-token training.

    The target sequence is the input sequence shifted one character to the
    right. For example, an input ``[a, b, c]`` has target ``[b, c, d]``. This
    lets the model learn to predict the next character at each time step.

    Args:
        vectorized_songs: One-dimensional NumPy array.
        seq_length: Number of time steps in each sampled input sequence.
        batch_size: Number of independent sequences to sample.

    Returns:
        A tuple (x_batch, y_batch)
    """
    # the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n - seq_length, batch_size)

    # Construct a list of input sequences for the training batch.
    input_batch = [vectorized_songs[i: i + seq_length] for i in idx]

    # Construct a list of output sequences for the training batch.
    output_batch = [vectorized_songs[i + 1: i + 1 + seq_length] for i in idx]

    # Convert the input and output batches to tensors
    x_batch = torch.tensor(input_batch, dtype=torch.long)
    y_batch = torch.tensor(output_batch, dtype=torch.long)

    return x_batch, y_batch

### Defining the RNN Model ###

class LSTMModel(nn.Module):
    """
    Character-level LSTM model for music generation.

    The model maps music characters to embeddings, processes them with an
    LSTM, and projects each time step to logits over the vocabulary. During
    training, those logits are compared with the next-character targets.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        """
        Initialize the embedding, recurrent, and output layers.

        Args:
            vocab_size: Number of unique characters in the vocabulary.
            embedding_dim: Size of each learned character embedding vector.
            hidden_size: Number of features in the LSTM hidden state.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        # Layer 1: Embedding layer to transform indices into dense vectors
        #   of a fixed embedding size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Layer 2: LSTM with hidden_size `hidden_size`.
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, batch_first=True)

        # Layer 3: Linear layer that transforms LSTM outputs to vocabulary logits.
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        """
        Create zero-initialized hidden and cell states for the LSTM.

        Args:
            batch_size: Number of sequences in the current batch.
            device: PyTorch device where the hidden states should be allocated.

        Returns:
            A tuple containing the initial hidden state and cell
            state with zeros
        """
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x, state=None, return_state=False):
        """
        Pass the input sequence through the LSTM model.

        Args:
            x: A batch of music characters.
            state: The hidden and cell state from the last step. If there is no
                state given, the model starts with zeros.
            return_state: Whether to also return the new LSTM state.

        Returns:
            The model's predictions for the next character at each position.
            If ``return_state`` is True, it also returns the final LSTM state.
        """
        x = self.embedding(x)

        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)

        out = self.fc(out)
        return out if not return_state else (out, state)

### Defining the loss function ###

cross_entropy = nn.CrossEntropyLoss() # instantiates the function
def compute_loss(labels, logits):
    """
    Inputs:
      labels: (batch_size, sequence_length)
      logits: (batch_size, sequence_length, vocab_size)

    Output:
      loss: scalar cross entropy loss over the batch and sequence length
    """

    # Batch the labels so that the shape of the labels should be (B * L,)
    batched_labels = labels.view(-1)

    '''Batch the logits so that the shape of the logits should be (B * L, V) '''
    batched_logits = logits.view(-1, logits.size(-1))

    '''Compute the cross-entropy loss using the batched  next characters and predictions'''
    loss = cross_entropy(batched_logits, batched_labels)
    return loss

### Hyperparameter setting and optimization ###

vocab_size = len(vocab)

# Model parameters:
params = dict(
  num_training_iterations = 1000, 
  batch_size = 64,  
  seq_length = 300,  
  learning_rate = 1e-3,  
  embedding_dim = 256,
  hidden_size = 1024,  
)

# Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

### Create a Comet experiment to track our training run ###

def create_experiment():
  # end any prior experiments
  if 'experiment' in locals():
    experiment.end()

  # initiate the comet experiment for tracking
  experiment = comet_ml.Experiment(
                  api_key=COMET_API_KEY,
                  project_name="6S191_Lab1_Part2")
  # log our hyperparameters, defined above, to the experiment
  for param, value in params.items():
    experiment.log_parameter(param, value)
  experiment.flush()

  return experiment

### Define optimizer and training operation ###

'''instantiate a new LSTMModel model for training using the hyperparameters
    created above.'''
model = LSTMModel(vocab_size, params["embedding_dim"], params["hidden_size"])

# Move the model to the GPU
model.to(device)

'''instantiate an optimizer with its learning rate.
  Checkout the PyTorch website for a list of supported optimizers.
  https://pytorch.org/docs/stable/optim.html
  Try using the Adam optimizer to start.'''
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

def train_step(x, y):
  # Set the model's mode to train
  model.train()

  # Zero gradients for every step
  optimizer.zero_grad()

  # Forward pass
  y_hat = model(x)

  # Compute the loss
  loss = compute_loss(y, y_hat)

  # Backward pass
  loss.backward()
  optimizer.step()

  return loss

##################
# Begin training!#
##################

history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
experiment = create_experiment()

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
for iter in tqdm(range(params["num_training_iterations"])):

    # Grab a batch and propagate it through the network
    x_batch, y_batch = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])

    # Convert numpy arrays to PyTorch tensors
    x_batch = torch.tensor(x_batch, dtype=torch.long).to(device)
    y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

    # Take a train step
    loss = train_step(x_batch, y_batch)

    # Log the loss to the Comet interface
    experiment.log_metric("loss", loss.item(), step=iter)

    # Update the progress bar and visualize within notebook
    history.append(loss.item())
    plotter.plot(history)

    # Save model checkpoint
    if iter % 100 == 0:
        torch.save(model.state_dict(), checkpoint_prefix)

# Save the final trained model
torch.save(model.state_dict(), checkpoint_prefix)
experiment.flush()

### Prediction of a generated song ###

def generate_text(model, start_string, generation_length=1000):
  # Evaluation step (generating ABC text using the learned RNN model)

  # convert the start string to numbers (vectorize)
  input_idx = [char2idx[s] for s in start_string] 
  input_idx = torch.tensor([input_idx], dtype=torch.long).to(device)

  # Initialize the hidden state
  state = model.init_hidden(input_idx.size(0), device)

  # Empty string to store our results
  text_generated = []
  tqdm._instances.clear()

  for i in tqdm(range(generation_length)):
    # evaluate the inputs and generate the next character predictions
    predictions, state = model(input_idx, state, return_state=True)

    # Remove the batch dimension
    predictions = predictions.squeeze(0)

    # use a multinomial distribution to sample over the probabilities
    input_idx = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1) 

    # add the predicted character to the generated text!
    # Hint: consider what format the prediction is in vs. the output
    text_generated.append(idx2char[input_idx].item()) 

  return (start_string + ''.join(text_generated))


generated_text = generate_text(model, start_string="X", generation_length=1000) 

### Play back generated songs ###

generated_songs = mdl.lab1.extract_song_snippet(generated_text)

for i, song in enumerate(generated_songs):
  # Synthesize the waveform from a song
  waveform = mdl.lab1.play_song(song)

  # If its a valid song (correct syntax), lets play it!
  if waveform:
    print("Generated song", i)
    ipythondisplay.display(waveform)

    numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
    wav_file_path = f"output_{i}.wav"
    write(wav_file_path, 88200, numeric_data)

    # save your song to the Comet interface -- you can access it there
    experiment.log_asset(wav_file_path)
    
experiment.end()
