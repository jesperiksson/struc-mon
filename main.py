# The top level program file
# Imports

# Libraries and modules
import argparse

# Other files
from ProgramRunner import ProgramRunner
from Settings import Settings
import config


def main(): 
    settings = Settings()
    # This parser allows the user to set flags 
    parser = argparse.ArgumentParser(
        description = '',
        epilog = '')
    parser.add_argument(
        '--q', 
        help = 'Dont show progress during training and evaluation',
        action = "store_true",
        #metavar = 'QUIET'
        )
    parser.add_argument(
        '--sensor',
        action = 'store',
        nargs = '*',
        default = [],
        type = str,
        help = 'Which kind of data to use. These are available(keyword-sensor): \nacc - Acceleration[g] \nincl - Inclinations[deg] \nstrain - Strain[mV]'
        )
    parser.add_argument(
        '--preset',
        action = 'store',
        nargs = 1,
        default = None,
        type = str,
        help = 'Uses a different preset neural net'
        )
    parser.add_argument(
        '--load',
        action='store_true',
        #nargs = 0,
        help = 'If provided the model tries to load an existing neural net')
    parser.add_argument(
        '--name',
        action = 'store',
        nargs = 1,
        default = None,
        type = str,
        #choices = , TODO: list exixting models
        help='Try to load a model with the current name (given the --load argument is provided)')
    args = parser.parse_args()
    ProgramRunner(settings,args)
            

if __name__ == "__main__":
    main()
