'''
Predicting the outcome of a tennis match using convolutional neural networks for regression

CNN trained to predict the final score (0-1, 2-1, etc.)

Input matrix is a 4x4:

| Player 1 Rank | Player 2 Rank | Avg. Service Speed P1 | Avg. Service Speed P2
| Surface wins ratio Player 1 (3 values)                | Time Elapsed since last match P1
| Surface wins ratio Player 2 (3 values)                | Time Elapsed since last match P2
| Court         | Surface       | Round                 | Series

Where Court, Surface, Round and Series are integers
'''

def prep_matrix(row):
    
