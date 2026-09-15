from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'runners'))
from module_folding_community_npu import main
if __name__ == '__main__':
    main('T-098')
