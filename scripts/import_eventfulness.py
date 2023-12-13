import sys
import os

def importParent():
    current = os.path.dirname(os.path.realpath(__file__))
    src = os.path.join(os.path.dirname(current), "eventfulness")
    sys.path.append(src)

