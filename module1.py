import os

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

def setup1():
    os.makedirs(POS_PATH)
    os.makedirs(NEG_PATH)
    os.makedirs(ANC_PATH)

    if os.path.isfile('lfw.tgz'):
       os.system('%tar -xf lfw.tgz')

    for directory in os.listdir('lfw'):
        for file in os.listdir(os.path.join('lfw', directory)):
            EX_PATH = os.path.join('lfw', directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)