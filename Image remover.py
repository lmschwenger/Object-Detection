#Remove images
import os
directory = r'D:/trainingImages/trainingImages/resistor/'

snp = 'snp'
dark = 'dark'
bright = 'bright'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if snp in f:
        os.remove(f)
    elif dark in f:
        os.remove(f)
    elif bright in f:
        os.remove(f)
