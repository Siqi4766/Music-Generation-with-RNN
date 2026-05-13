# Music-Generation-with-RNN
This project trains a recurrent neural network to generate music in ABC notation. It is adapted from MIT Introduction to Deep Learning Lab 1, Part 2: Music Generation with RNNs.

The model learns from a dataset of Irish folk songs represented as text. Each character in the ABC notation is converted into a number, and the model is trained to predict the next character in a sequence. After training, the model generates new ABC text one character at a time, and valid generated songs are converted into `.wav` audio files.

## Project Source

Adapted from: https://github.com/MITDeepLearning/introtodeeplearning/blob/master/lab1/PT_Part2_Music_Generation.ipynb

## Training Settings

For my run, I used the following hyperparameters:

| Parameter | Value |
| --- | --- |
| Training iterations | 1000 |
| Batch size | 64 |
| Sequence length | 300 |
| Learning rate | 0.001 |
| Embedding dimension | 256 |
| Hidden size | 1024 |

The model was trained with the Adam optimizer and cross-entropy loss.

## Results

In this run:

- The recorded loss values ranged from about `4.43` to `0.084`.
- The generated ABC text contained 3 valid song snippets.
- The valid generated songs were saved as `.wav` files.

Example output files:

- `output_0.wav`
- `output_1.wav`
- `output_2.wav`

