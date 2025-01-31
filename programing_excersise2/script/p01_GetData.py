class main():
    def __init__(self, pathes):
        self.pathes = pathes
        self.sequences = []
        self.length = []




    def read_fasta_file(self):
        import math
        from Bio import SeqIO
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        for path in self.pathes:
            sequences = []
            for record in SeqIO.parse(path, "fasta"):
                sequences.append(record.seq)
            self.sequences.append(sequences)

            length = []
            for sequences in self.sequences:
                for sequence in sequences:
                    length.append(len(sequence))
            self.length.append(length)