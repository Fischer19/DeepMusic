import os
import numpy as np
import pickle as pl
import random

PATTERN_LEN=5

def generate_sentence():
    note_sequence_length=random.randint(7,11)
    pattern=random.randint(0,1)
    train_X=[]
    train_Y=[]
    end_note=0
    lasting=0
    previous_note=-1
    for i in range(note_sequence_length):
        if((i+PATTERN_LEN-1)<(sequence_length-1)):
            lasting=random.randint()
