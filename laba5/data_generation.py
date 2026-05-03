import numpy as np



def generate_data():
    np.random.seed(42)
    X = np.random.randint(0,2,size=(1000,12))
    y = np.random.randint(0,2,size=(1000,2))

    np.savetxt("laba5/dataIn.txt", X, fmt="%d")
    np.savetxt("laba5/dataOut.txt", y, fmt="%d")