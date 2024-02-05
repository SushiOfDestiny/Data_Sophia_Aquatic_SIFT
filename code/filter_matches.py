import sys

# add ../blender to path
sys.path.append('../blender')

from blender.matching import (
    check_correct_match,
    save_correct_matches,
)

if __name__ == "__main__":
# Goal is to load the saved opencv matches and to filter them with the blender script


